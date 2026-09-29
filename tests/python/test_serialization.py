# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Tests for saving and loading models."""

from __future__ import annotations

import contextlib
import copy
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rclib
from rclib import ESN, model, readouts, reservoirs

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Reservoir layouts by name: (connection type, reservoir configurations).
TOPOLOGIES: dict[str, tuple[str, list[reservoirs.RandomSparse | reservoirs.Nvar]]] = {
    "random_sparse": (
        "serial",
        [reservoirs.RandomSparse(n_neurons=30, spectral_radius=0.9, include_bias=True, seed=3)],
    ),
    "nvar": ("serial", [reservoirs.Nvar(num_lags=3, polynomial_order=2)]),
    "serial_mixed": (
        "serial",
        [
            reservoirs.RandomSparse(n_neurons=8, spectral_radius=0.9, seed=3),
            reservoirs.Nvar(num_lags=2, polynomial_order=2),
        ],
    ),
    "parallel_mixed": (
        "parallel",
        [
            reservoirs.RandomSparse(n_neurons=30, spectral_radius=0.9, seed=3),
            reservoirs.Nvar(num_lags=2, polynomial_order=2),
        ],
    ),
}
READOUTS: dict[str, readouts.Ridge | readouts.Rls | readouts.Lms] = {
    "ridge": readouts.Ridge(alpha=1e-4, include_bias=True),
    "ridge_cg_implicit": readouts.Ridge(
        alpha=1e-4, include_bias=False, solver="conjugate_gradient_implicit", tolerance=1e-9
    ),
    "rls_rank1": readouts.Rls(lambda_=0.99, delta=1.0, include_bias=True),
    # The rank-k (Woodbury) update only runs for lambda == 1 and batches of more than one row.
    "rls_rank_k": readouts.Rls(lambda_=1.0, delta=1.0, include_bias=True, solver="rank_k_update"),
    "lms": readouts.Lms(learning_rate=0.01, include_bias=True),
}
ONLINE_READOUTS = ["rls_rank1", "rls_rank_k", "lms"]


def _signal(n_samples: int, phase: float = 0.0) -> np.ndarray:
    """One-column series whose next value is the prediction target."""
    return np.sin(0.3 * np.arange(n_samples) + phase).reshape(-1, 1)


def _build(topology: str, readout: str) -> ESN:
    connection_type, configs = TOPOLOGIES[topology]
    esn = ESN(connection_type)
    for config in configs:
        esn.add_reservoir(config)
    esn.set_readout(READOUTS[readout])
    return esn


def _fitted(topology: str, readout: str) -> ESN:
    esn = _build(topology, readout)
    series = _signal(61)
    esn.fit(series[:-1], series[1:], washout_len=5)
    return esn


def _configs(esn: ESN) -> list[tuple[type, dict[str, object]]]:
    """The reservoir and readout configurations as comparable (type, attributes) pairs."""
    params = [*esn._reservoirs_params, esn._readout_params]  # noqa: SLF001
    return [(type(config), vars(config)) for config in params]


@pytest.mark.parametrize("readout", READOUTS)
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_round_trip_predicts_identically(tmp_path: Path, topology: str, readout: str) -> None:
    """A loaded model predicts bit-identically and has the same configuration."""
    original = _fitted(topology, readout)
    path = tmp_path / "model.rclib"
    original.save(path)

    restored = ESN.load(path)
    assert restored.connection_type == original.connection_type
    assert _configs(restored) == _configs(original)
    probe = _signal(15, phase=1.0)
    np.testing.assert_array_equal(restored.predict(probe), original.predict(probe))


def test_paths_can_be_str_or_path_like(tmp_path: Path) -> None:
    """Save and load accept both str and os.PathLike paths."""
    original = _fitted("random_sparse", "ridge")
    original.save(str(tmp_path / "a.rclib"))
    original.save(tmp_path / "b.rclib")
    probe = _signal(5)
    for restored in (ESN.load(tmp_path / "a.rclib"), ESN.load(str(tmp_path / "b.rclib"))):
        np.testing.assert_array_equal(restored.predict(probe), original.predict(probe))


@pytest.mark.parametrize(
    "solver", ["auto", "cholesky", "dual_cholesky", "conjugate_gradient", "conjugate_gradient_implicit"]
)
def test_ridge_solver_names_round_trip(tmp_path: Path, solver: str) -> None:
    """Every Ridge solver name survives a round trip, also before fitting."""
    esn = ESN()
    esn.add_reservoir(reservoirs.Nvar(num_lags=2))
    esn.set_readout(readouts.Ridge(alpha=0.5, include_bias=True, solver=solver, tolerance=1e-7))
    esn.save(tmp_path / "model.rclib")
    assert _configs(ESN.load(tmp_path / "model.rclib")) == _configs(esn)


@pytest.mark.parametrize("readout", ONLINE_READOUTS)
@pytest.mark.parametrize("topology", ["random_sparse", "serial_mixed", "parallel_mixed"])
def test_online_learning_continues_identically(tmp_path: Path, topology: str, readout: str) -> None:
    """Saved reservoir states and readout state let online learning continue exactly."""
    original = _build(topology, readout)
    series = _signal(61)
    original.fit(series[:20], series[1:21])
    original.save(tmp_path / "model.rclib")
    restored = ESN.load(tmp_path / "model.rclib")

    # Two-row batches, then single steps that update from the current state (x=None).
    for start in range(20, 40, 2):
        x, y = series[start : start + 2], series[start + 1 : start + 3]
        np.testing.assert_array_equal(restored.predict_online(x), original.predict_online(x))
        restored.partial_fit(x, y)
        original.partial_fit(x, y)
    for step in range(40, 60):
        x, y = series[step : step + 1], series[step + 1 : step + 2]
        np.testing.assert_array_equal(restored.predict_online(x), original.predict_online(x))
        restored.partial_fit(None, y)
        original.partial_fit(None, y)
    np.testing.assert_array_equal(restored.predict(series), original.predict(series))


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_generative_prediction_continues_identically(tmp_path: Path, topology: str) -> None:
    """Without priming input, generation starts from the saved reservoir states."""
    original = _fitted(topology, "ridge")
    original.save(tmp_path / "model.rclib")
    restored = ESN.load(tmp_path / "model.rclib")
    no_priming = np.empty((0, 1))
    np.testing.assert_array_equal(
        restored.predict_generative(no_priming, 10), original.predict_generative(no_priming, 10)
    )


@pytest.mark.parametrize("readout", ["rls_rank1", "lms"])
def test_readout_left_unfitted_by_a_failed_fit(tmp_path: Path, readout: str) -> None:
    """A readout emptied by a failed fit loads unfitted and restarts identically."""
    original = _build("random_sparse", readout)
    series = _signal(41)
    original.fit(series[:20], series[1:21])
    # RLS rejects an empty batch and LMS accepts it; both end up unfitted.
    with contextlib.suppress(ValueError):
        original._cpp_model.getReadout().fit(np.empty((0, 30)), np.empty((0, 1)))  # noqa: SLF001
    original.save(tmp_path / "model.rclib")
    restored = ESN.load(tmp_path / "model.rclib")

    # predict advances the reservoirs before the readout raises, so call it on both.
    for esn in (restored, original):
        with pytest.raises(RuntimeError, match="must be fit"):
            esn.predict(series)
    for start in range(20, 40, 2):
        restored.partial_fit(series[start : start + 2], series[start + 1 : start + 3])
        original.partial_fit(series[start : start + 2], series[start + 1 : start + 3])
    np.testing.assert_array_equal(restored.predict(series), original.predict(series))


def test_serialization_error_is_a_runtime_error() -> None:
    """SerializationError is exported and subclasses RuntimeError."""
    assert issubclass(rclib.SerializationError, RuntimeError)


@pytest.mark.parametrize("contents", [b"", b"not a model file", "truncated"])
def test_invalid_files_raise(tmp_path: Path, contents: bytes | str) -> None:
    """Invalid, empty and truncated files raise SerializationError."""
    path = tmp_path / "model.rclib"
    if contents == "truncated":
        _fitted("random_sparse", "ridge").save(path)
        contents = path.read_bytes()[:-1]
    assert isinstance(contents, bytes)
    path.write_bytes(contents)
    with pytest.raises(rclib.SerializationError):
        ESN.load(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    """Loading a missing file raises SerializationError."""
    with pytest.raises(rclib.SerializationError, match="cannot open"):
        ESN.load(tmp_path / "missing.rclib")


def _without_readout() -> ESN:
    esn = ESN()
    esn.add_reservoir(reservoirs.Nvar(num_lags=2))
    return esn


def _without_reservoir() -> ESN:
    esn = ESN()
    esn.set_readout(readouts.Ridge(alpha=1.0, include_bias=True))
    return esn


def _with_mismatched_readout() -> ESN:
    esn = ESN()
    esn.add_reservoir(reservoirs.RandomSparse(n_neurons=5, spectral_radius=0.9))
    esn.set_readout(readouts.Ridge(alpha=1.0, include_bias=True))
    esn.get_reservoir(0).advance(np.ones((1, 1)))  # locks the reservoir to 5 outputs
    rng = np.random.default_rng(seed=0)
    esn._cpp_model.getReadout().fit(rng.random((10, 7)), rng.random((10, 1)))  # noqa: SLF001
    return esn


def _with_mismatched_readout_after_unused_reservoir() -> ESN:
    # A RandomSparse reservoir outputs n_neurons even before its first input.
    esn = ESN()
    esn.add_reservoir(reservoirs.RandomSparse(n_neurons=5, spectral_radius=0.9))
    esn.set_readout(readouts.Ridge(alpha=1.0, include_bias=True))
    rng = np.random.default_rng(seed=0)
    esn._cpp_model.getReadout().fit(rng.random((10, 7)), rng.random((10, 1)))  # noqa: SLF001
    return esn


@pytest.mark.parametrize(
    "make_model",
    [_without_readout, _without_reservoir, _with_mismatched_readout, _with_mismatched_readout_after_unused_reservoir],
)
def test_rejected_save_keeps_the_existing_file(tmp_path: Path, make_model: Callable[[], ESN]) -> None:
    """A model that cannot be saved raises and leaves an existing file untouched."""
    path = tmp_path / "model.rclib"
    _fitted("random_sparse", "ridge").save(path)
    checkpoint = path.read_bytes()

    with pytest.raises(rclib.SerializationError):
        make_model().save(path)
    assert path.read_bytes() == checkpoint
    assert [entry.name for entry in tmp_path.iterdir()] == ["model.rclib"]


def test_config_rebuild_errors_raise_serialization_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors while rebuilding the Python configuration surface as SerializationError."""
    _fitted("random_sparse", "ridge").save(tmp_path / "model.rclib")

    def reject(_cpp_readout: object) -> readouts.Ridge:
        msg = "alpha must be finite and non-negative."
        raise ValueError(msg)

    monkeypatch.setattr(model, "_readout_config_from_cpp", reject)
    with pytest.raises(rclib.SerializationError, match="alpha"):
        ESN.load(tmp_path / "model.rclib")


@pytest.mark.parametrize("topology", ["random_sparse", "parallel_mixed"])
def test_pickle_round_trip(tmp_path: Path, topology: str) -> None:
    """Pickling stores the same bytes as a model file and restores an identical model."""
    original = _fitted(topology, "rls_rank1")
    original.save(tmp_path / "model.rclib")
    assert original._cpp_model.dumps() == (tmp_path / "model.rclib").read_bytes()  # noqa: SLF001

    restored = pickle.loads(pickle.dumps(original))  # noqa: S301 - data pickled by this test
    assert restored.connection_type == original.connection_type
    assert _configs(restored) == _configs(original)
    x = _signal(4, phase=0.5)
    np.testing.assert_array_equal(restored.predict_online(x), original.predict_online(x))


def test_deepcopy_is_independent() -> None:
    """copy.deepcopy yields an identical model that no longer shares state with the original."""
    original = _fitted("random_sparse", "rls_rank1")
    duplicate = copy.deepcopy(original)
    series = _signal(20, phase=0.5)
    np.testing.assert_array_equal(duplicate.predict(series), original.predict(series))

    before = original.predict(series)
    duplicate.partial_fit(series[:-1], series[1:])
    np.testing.assert_array_equal(original.predict(series), before)
    assert not np.array_equal(duplicate.predict(series), before)


class _TaggedESN(ESN):
    """An ESN subclass with state of its own, as users might define."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag


def _pickle_round_trip(esn: ESN) -> ESN:
    return pickle.loads(pickle.dumps(esn))  # noqa: S301 - data pickled by this test


@pytest.mark.parametrize("duplicate", [copy.deepcopy, _pickle_round_trip], ids=["deepcopy", "pickle"])
def test_copies_keep_instance_state_and_subclass(duplicate: Callable[[ESN], ESN]) -> None:
    """Attributes added by users or subclasses, and the subclass itself, survive copying."""
    original = _TaggedESN("baseline")
    original.add_reservoir(reservoirs.RandomSparse(n_neurons=20, spectral_radius=0.9, seed=3))
    original.set_readout(readouts.Ridge(alpha=1e-4, include_bias=True))
    series = _signal(41)
    original.fit(series[:-1], series[1:], washout_len=5)
    original.training_metadata = {"dataset": "example"}

    restored = duplicate(original)
    assert type(restored) is _TaggedESN
    assert restored.tag == "baseline"
    assert restored.training_metadata == {"dataset": "example"}
    assert restored.training_metadata is not original.training_metadata
    assert _configs(restored) == _configs(original)
    np.testing.assert_array_equal(restored.predict(series), original.predict(series))


def test_pickling_an_unsavable_model_raises() -> None:
    """Pickling a model that cannot be saved raises SerializationError."""
    with pytest.raises(rclib.SerializationError):
        pickle.dumps(_without_readout())
