# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Tests for the configurations of reward-modulated Hebbian learning."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest
from rclib import _rclib, learning, readouts
from rclib.model import _rule_from_cpp, _rule_to_cpp

NAN = math.nan
INF = math.inf


def test_defaults_are_valid_and_compare_by_value() -> None:
    """Configurations are frozen values: equal settings compare equal."""
    rule = learning.RewardModulatedHebbian(learning_rate=0.1)
    assert rule == learning.RewardModulatedHebbian(learning_rate=0.1)
    assert rule != learning.RewardModulatedHebbian(learning_rate=0.2)
    assert rule.exploration == learning.Exploration()
    assert rule.baseline == learning.Baseline()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.learning_rate = 0.5
    with pytest.raises(TypeError):
        learning.RewardModulatedHebbian(0.1)  # keyword-only


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"distribution": "laplace"}, "distribution"),
        ({"amplitude": -1.0}, "amplitude"),
        ({"amplitude": NAN}, "amplitude"),
        ({"amplitude": INF}, "amplitude"),
        ({"probability": 0.0}, "probability"),
        ({"probability": 1.5}, "probability"),
        ({"probability": NAN}, "probability"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
    ],
)
def test_exploration_rejects_invalid_settings(kwargs: dict[str, Any], match: str) -> None:
    """Exploration settings are checked as the C++ core checks them."""
    with pytest.raises(ValueError, match=match):
        learning.Exploration(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "median"}, "kind"),
        ({"decay": 1.0}, "decay"),
        ({"decay": -0.1}, "decay"),
        ({"decay": NAN}, "decay"),
        ({"window": 0}, "window"),
        ({"window": 1_000_001}, "window"),
    ],
)
def test_baseline_rejects_invalid_settings(kwargs: dict[str, Any], match: str) -> None:
    """Baseline settings are checked as the C++ core checks them."""
    with pytest.raises(ValueError, match=match):
        learning.Baseline(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": NAN}, "learning_rate"),
        ({"learning_rate": INF}, "learning_rate"),
        ({"post": "rate"}, "post"),
        ({"post_decay": 1.0}, "post_decay"),
        ({"post_decay": NAN}, "post_decay"),
        ({"eligibility_power": 0.0}, "eligibility_power"),
        ({"eligibility_power": INF}, "eligibility_power"),
        ({"update": "batch"}, "update"),
        ({"trace_decay": 1.5}, "trace_decay"),
        ({"trace_decay": NAN}, "trace_decay"),
        ({"update": "trial", "trace_decay": 0.5}, "trace_decay"),
        ({"modulator": "sign"}, "modulator"),
        ({"learning_rate_decay_steps": 0.0}, "learning_rate_decay_steps"),
        ({"learning_rate_decay_steps": NAN}, "learning_rate_decay_steps"),
        ({"max_update": -1.0}, "max_update"),
        ({"decorrelation": 1.0}, "decorrelation requires trial"),
        ({"update": "trial", "decorrelation": 0.0}, "decorrelation"),
        ({"update": "trial", "decorrelation": 1.0, "eligibility_power": 3.0}, "eligibility_power of 1"),
    ],
)
def test_rule_rejects_invalid_settings(kwargs: dict[str, Any], match: str) -> None:
    """Rule settings and their combinations are checked as the C++ core checks them."""
    settings = {"learning_rate": 0.1, **kwargs}
    with pytest.raises(ValueError, match=match):
        learning.RewardModulatedHebbian(**settings)


@pytest.mark.parametrize(("field", "value"), [("exploration", 0.1), ("baseline", "ema")])
def test_rule_rejects_parts_of_the_wrong_type(field: str, value: Any) -> None:  # noqa: ANN401
    """Exploration and baseline must be their configuration classes."""
    with pytest.raises(TypeError, match=field):
        learning.RewardModulatedHebbian(learning_rate=0.1, **{field: value})


RULES = [
    learning.RewardModulatedHebbian(learning_rate=0.1),
    learning.RewardModulatedHebbian(
        learning_rate=2.5,
        exploration=learning.Exploration(distribution="gaussian", amplitude=0.05, probability=0.25, seed=2**32 - 1),
        post="noise",
        eligibility_power=3.0,
        update="online",
        trace_decay=0.9,
        modulator="heaviside",
        baseline=learning.Baseline(kind="window", window=7),
        learning_rate_decay_steps=100.0,
        max_update=1e-3,
    ),
    learning.RewardModulatedHebbian(
        learning_rate=0.5,
        post="potential",
        update="trial",
        baseline=learning.Baseline(kind="none"),
        decorrelation=1.0,
    ),
]


@pytest.mark.parametrize("rule", RULES)
def test_rule_round_trips_through_the_cpp_rule(rule: learning.RewardModulatedHebbian) -> None:
    """Every setting reaches the C++ rule and comes back unchanged."""
    cpp_rule = _rule_to_cpp(rule)
    cpp_rule.validate()
    assert _rule_from_cpp(cpp_rule) == rule


def test_reward_modulated_readout_config() -> None:
    """The readout configuration checks its output count and rule type."""
    rule = learning.RewardModulatedHebbian(learning_rate=0.1)
    readout = readouts.RewardModulated(3, rule, include_bias=False)
    assert (readout.n_outputs, readout.rule, readout.include_bias) == (3, rule, False)
    with pytest.raises(ValueError, match="n_outputs"):
        readouts.RewardModulated(0, rule, include_bias=True)
    with pytest.raises(TypeError, match="rule"):
        readouts.RewardModulated(1, _rclib.RewardModulatedHebbianRule(), include_bias=True)
