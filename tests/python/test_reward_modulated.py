# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Tests for reward-modulated Hebbian learning against a NumPy reference of the rule."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rclib import ESN, _rclib, learning, readouts, reservoirs
from rclib.model import _rule_to_cpp

# Results of the reference agree with the C++ core up to the summation order of products.
RTOL = 1e-10
ATOL = 1e-12


def _eligibility_function(values: np.ndarray, power: float) -> np.ndarray:
    """S(y) = sign(y) |y|^power."""
    return values if power == 1 else np.sign(values) * np.abs(values) ** power


class _ReferenceLearner:
    """The rule of rclib.learning applied to weights of shape (n_pre, n_post), with given noise."""

    def __init__(self, rule: learning.RewardModulatedHebbian, weights: np.ndarray) -> None:
        self.rule = rule
        self.weights = weights.copy()
        n_pre, n_post = weights.shape
        self.eligibility = np.zeros((n_pre, n_post))
        self.covariance = np.zeros((n_pre, n_pre))
        self.average: np.ndarray | None = None
        self.baselines: dict[int, list[float]] = {}
        self.updates = 0

    def row(self, pre: np.ndarray, potential: np.ndarray, noise: np.ndarray) -> None:
        rule = self.rule
        if rule.post == "noise":
            post = noise
        elif rule.post == "potential":
            post = potential
        else:
            previous = potential if self.average is None else self.average
            post = potential - previous
            self.average = rule.post_decay * previous + (1 - rule.post_decay) * potential
        power = rule.eligibility_power
        term = np.outer(_eligibility_function(pre, power), _eligibility_function(post, power))
        if rule.update == "online":
            self.eligibility = rule.trace_decay * self.eligibility + term
        else:
            self.eligibility = self.eligibility + term
            if rule.decorrelation is not None:
                self.covariance = self.covariance + np.outer(pre, pre)

    def reset_activity(self) -> None:
        self.average = None

    def reinforce(self, reward: float, key: int, baseline: float | None) -> None:
        rule = self.rule
        history = self.baselines.get(key)
        new_history: list[float] | None = None
        if baseline is not None:
            expected = baseline
        elif rule.baseline.kind == "none":
            expected = 0.0
        elif rule.baseline.kind == "ema":
            expected = reward if history is None else history[0]
            new_history = [rule.baseline.decay * expected + (1 - rule.baseline.decay) * reward]
        else:
            expected = reward if not history else sum(value / len(history) for value in history)
            new_history = [*(history or []), reward][-rule.baseline.window :]
        if rule.modulator == "difference":
            modulator = reward - expected
        else:
            modulator = 1.0 if reward > expected else 0.0
        self.modulate(modulator)
        if new_history is not None:
            self.baselines[key] = new_history

    def modulate(self, modulator: float) -> None:
        rule = self.rule
        if modulator != 0:
            direction = self.eligibility
            if rule.decorrelation is not None:
                regularized = self.covariance + rule.decorrelation * np.eye(len(self.covariance))
                direction = np.linalg.solve(regularized, self.eligibility)
            eta = rule.learning_rate
            if rule.learning_rate_decay_steps is not None:
                eta = rule.learning_rate / (1 + self.updates / rule.learning_rate_decay_steps)
            delta = (eta * modulator) * direction
            if rule.max_update is not None:
                delta = np.clip(delta, -rule.max_update, rule.max_update)
            self.weights = self.weights + delta
        self.updates += 1
        if rule.update == "trial":
            self.eligibility = np.zeros_like(self.eligibility)
            self.covariance = np.zeros_like(self.covariance)


class _ReferenceReadout(_ReferenceLearner):
    """A reward-modulated readout: the potential is ``[x, 1] @ W + noise``."""

    def __init__(self, rule: learning.RewardModulatedHebbian, weights: np.ndarray, *, include_bias: bool) -> None:
        super().__init__(rule, weights)
        self.include_bias = include_bias

    def act(self, states: np.ndarray, noise: np.ndarray) -> np.ndarray:
        outputs = []
        for state, row_noise in zip(states, noise, strict=True):
            pre = np.append(state, 1.0) if self.include_bias else state
            potential = pre @ self.weights + row_noise
            self.row(pre, potential, row_noise)
            outputs.append(potential)
        return np.array(outputs)


def _rule(**kwargs: object) -> learning.RewardModulatedHebbian:
    return learning.RewardModulatedHebbian(learning_rate=0.3, **kwargs)  # type: ignore[arg-type]


ORACLE_RULES = {
    "eh_high_pass_ema": _rule(post="high_pass", post_decay=0.0),
    "eh_slow_filter": _rule(post="high_pass", post_decay=0.8, baseline=learning.Baseline(decay=0.5)),
    "noise_window": _rule(post="noise", baseline=learning.Baseline(kind="window", window=3)),
    "potential_no_baseline": _rule(post="potential", baseline=learning.Baseline(kind="none")),
    "heaviside_decay": _rule(post="high_pass", modulator="heaviside", learning_rate_decay_steps=2.0),
    "online_trace": _rule(post="noise", trace_decay=0.6, max_update=0.05),
    "trial_cube": _rule(post="high_pass", post_decay=0.3, eligibility_power=3.0, update="trial", max_update=0.1),
    "trial_signed_square": _rule(post="noise", eligibility_power=2.0, update="trial"),
    "trial_decorrelated": _rule(post="noise", update="trial", decorrelation=0.5),
    "trial_decorrelated_heaviside": _rule(
        post="noise",
        update="trial",
        decorrelation=2.0,
        modulator="heaviside",
        baseline=learning.Baseline(kind="window"),
    ),
}


@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("name", ORACLE_RULES)
def test_readout_follows_the_reference_rule(name: str, *, include_bias: bool) -> None:
    """Random sequences of rows, rewards, baselines and resets give the reference's weights."""
    rule = ORACLE_RULES[name]
    rng = np.random.default_rng(seed=sorted(ORACLE_RULES).index(name))
    n_features, n_outputs = 3, 2
    weights = rng.uniform(-1, 1, (n_features + include_bias, n_outputs))
    readout = _rclib.RewardModulatedReadout(n_outputs, _rule_to_cpp(rule), include_bias)
    readout.setWeights(weights)
    reference = _ReferenceReadout(rule, weights, include_bias=include_bias)

    for _ in range(4):
        for step in range(3):
            rows = int(rng.integers(1, 3))
            states = rng.uniform(-1, 1, (rows, n_features))
            noise = 0.1 * rng.standard_normal((rows, n_outputs))
            np.testing.assert_allclose(
                readout.actWithNoise(states, noise), reference.act(states, noise), rtol=RTOL, atol=ATOL
            )
            if rule.update == "online" or step == 2:  # noqa: PLR2004
                reward = float(rng.normal())
                key = int(rng.integers(0, 2))
                baseline = float(rng.normal()) if rng.random() < 0.2 else None  # noqa: PLR2004
                readout.reinforce(reward, key, baseline)
                reference.reinforce(reward, key, baseline)
                np.testing.assert_allclose(readout.getWeights(), reference.weights, rtol=RTOL, atol=ATOL)
        readout.resetActivity()
        reference.reset_activity()

    readout.actWithNoise(np.ones((1, n_features)), np.full((1, n_outputs), 0.05))
    reference.act(np.ones((1, n_features)), np.full((1, n_outputs), 0.05))
    readout.modulate(-0.7)
    reference.modulate(-0.7)
    np.testing.assert_allclose(readout.getWeights(), reference.weights, rtol=RTOL, atol=ATOL)
    learning_state = readout.getLearning()
    assert learning_state.getUpdateCount() == reference.updates
    assert learning_state.getBaselines().keys() == reference.baselines.keys()
    for key, history in reference.baselines.items():
        np.testing.assert_allclose(learning_state.getBaselines()[key], history, rtol=RTOL, atol=ATOL)


def _reward_esn(rule: learning.RewardModulatedHebbian | None = None, *, n_outputs: int = 2) -> ESN:
    esn = ESN()
    esn.add_reservoir(reservoirs.RandomSparse(n_neurons=20, spectral_radius=0.9, leak_rate=0.5, seed=1))
    rule = rule or _rule(post="noise", baseline=learning.Baseline(kind="none"))
    esn.set_readout(readouts.RewardModulated(n_outputs, rule, include_bias=True))
    return esn


def test_esn_act_reinforce_modulate_and_clear() -> None:
    """ESN.act explores through the readout and ESN.reinforce and ESN.modulate update it."""
    esn = _reward_esn()
    outputs = esn.act(np.ones((3, 1)))
    assert outputs.shape == (3, 2)
    readout = esn.get_readout()
    assert isinstance(readout, _rclib.RewardModulatedReadout)
    eligibility = readout.getLearning().getEligibility()
    esn.reinforce(2.0)
    np.testing.assert_allclose(readout.getWeights(), 0.3 * 2.0 * eligibility, rtol=1e-15)
    with pytest.raises(RuntimeError, match="exploring act"):
        esn.reinforce(1.0)
    esn.act(np.ones((1, 1)))
    esn.modulate(-1.0)
    assert readout.getLearning().getUpdateCount() == 2  # noqa: PLR2004
    esn.act(np.ones((1, 1)))
    esn.clear_eligibility()
    assert not readout.getLearning().hasEligibility()


def test_esn_act_rejects_invalid_input_and_rewards() -> None:
    """Non-finite inputs and rewards are ValueErrors, misuse is a RuntimeError."""
    esn = _reward_esn()
    with pytest.raises(ValueError, match="finite"):
        esn.act(np.array([[math.nan]]))
    with pytest.raises(RuntimeError, match="exploring act"):
        esn.reinforce(1.0)
    esn.act(np.ones((1, 1)))
    with pytest.raises(ValueError, match="reward must be finite"):
        esn.reinforce(math.inf)
    with pytest.raises(ValueError, match="baseline must be finite"):
        esn.reinforce(1.0, baseline=math.nan)
    with pytest.raises(RuntimeError, match="learns from rewards"):
        esn.fit(np.ones((5, 1)), np.ones((5, 2)))
    with pytest.raises(RuntimeError, match="learns from rewards"):
        esn.partial_fit(np.ones((1, 1)), np.ones((1, 2)))


def test_noise_free_act_equals_predict_online() -> None:
    """A noise-free act changes no learning state and gives predict_online's output bit for bit."""
    acting, predicting = _reward_esn(), _reward_esn()
    weights = np.random.default_rng(seed=2).uniform(-1, 1, (21, 2))
    for esn in (acting, predicting):
        esn.get_readout().setWeights(weights)
    inputs = np.sin(np.linspace(0, 3, 8)).reshape(-1, 1)
    np.testing.assert_array_equal(acting.act(inputs, explore=False), predicting.predict_online(inputs))
    assert not acting.get_readout().getLearning().hasEligibility()


def test_act_without_a_readout_returns_the_reservoir_states() -> None:
    """A model without a readout acts with its reservoir states, but has nothing to reinforce."""
    esn = ESN()
    esn.add_reservoir(reservoirs.RandomSparse(n_neurons=6, spectral_radius=0.9, seed=3))
    states = esn.act(np.ones((2, 1)))
    assert states.shape == (2, 6)
    np.testing.assert_array_equal(states[-1:], esn.get_reservoir(0).getState())
    with pytest.raises(RuntimeError, match="no reward-modulated component"):
        esn.reinforce(1.0)


def test_set_readout_swaps_a_supervised_readout_for_a_reward_modulated_one() -> None:
    """Supervised pre-training can hand its weights to a reward-modulated readout."""
    esn = ESN()
    esn.add_reservoir(reservoirs.RandomSparse(n_neurons=20, spectral_radius=0.9, leak_rate=0.5, seed=1))
    esn.set_readout(readouts.Ridge(alpha=1e-6, include_bias=True))
    rng = np.random.default_rng(seed=4)
    esn.fit(rng.random((50, 1)), rng.random((50, 2)))
    weights = esn.get_readout().getWeights()
    probe = rng.random((5, 1))
    expected = esn.predict(probe)

    esn.set_readout(readouts.RewardModulated(2, _rule(), include_bias=True))
    esn.get_readout().setWeights(weights)
    np.testing.assert_allclose(esn.predict(probe), expected, rtol=1e-12, atol=1e-12)


def test_readout_learns_a_linear_map_from_rewards_alone() -> None:
    """Node perturbation through an identity NVAR learns a 5-to-2 linear map from rewards."""
    esn = ESN()
    esn.add_reservoir(reservoirs.Nvar(num_lags=1, polynomial_order=1))  # passes the input through
    rule = learning.RewardModulatedHebbian(
        learning_rate=7.5, exploration=learning.Exploration(amplitude=0.1, seed=3), post="noise"
    )
    esn.set_readout(readouts.RewardModulated(2, rule, include_bias=True))
    esn.get_readout().setWeights(np.zeros((6, 2)))
    rng = np.random.default_rng(seed=5)
    mapping = rng.uniform(-1, 1, (5, 2))
    probe = rng.uniform(-1, 1, (200, 5))

    def error() -> float:
        return float(np.mean((esn.act(probe, explore=False) - probe @ mapping) ** 2))

    initial = error()
    for _ in range(3000):
        x = rng.uniform(-1, 1, (1, 5))
        target = x @ mapping
        clean = -float(np.sum((esn.act(x, explore=False) - target) ** 2))
        reward = -float(np.sum((esn.act(x) - target) ** 2))
        esn.reinforce(reward, baseline=clean)
    assert error() < initial / 10
