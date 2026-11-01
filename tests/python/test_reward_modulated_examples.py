# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Slow acceptance checks of the reward-modulated examples, at reduced size.

Each check runs an example's own task over the declared evaluation seeds. The compared arms
share the network seed, the exploration seed, the task sequence and the training budget, and
the claimed arm must win on at least 4 of the 5 seeds and on the median by a declared margin.
The outcomes are properties of these demonstrations, not general guarantees.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "python"
SEEDS = (0, 1, 2, 3, 4)
MIN_WINS = 4

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _load_example(name: str) -> ModuleType:
    """Import an example script as a module without running its main()."""
    spec = importlib.util.spec_from_file_location(f"rclib_example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_lower(claimed: list[float], other: list[float], margin: float) -> None:
    """The claimed arm's errors are lower on at least MIN_WINS seeds and lower by `margin` on the median."""
    wins = sum(c < o for c, o in zip(claimed, other, strict=True))
    assert wins >= MIN_WINS, (claimed, other)
    assert float(np.median(claimed)) + margin < float(np.median(other)), (claimed, other)


def test_pattern_generation_needs_exploration() -> None:
    """Hoerzer et al.: the exploring readout generates the sine; without noise it learns nothing."""
    example = _load_example("reward_modulated_pattern_generation")
    train_steps = 60_000
    errors = {}
    for name, amplitude in (("exploring", 0.2), ("no exploration", 0.0)):
        errors[name] = []
        for seed in SEEDS:
            esn = example.build_model(seed, exploration_amplitude=amplitude)
            example.train(esn, train_steps)
            errors[name].append(example.normalized_error(example.generate(esn, 500), train_steps))
    _assert_lower(errors["exploring"], errors["no exploration"], margin=0.5)


def test_cursor_control_recovers_and_assigns_credit() -> None:
    """Legenstein et al.: the EH rule recovers the perturbed decoder and shifts the rotated units more."""
    example = _load_example("reward_modulated_bci")
    final_rewards: dict[str, list[float]] = {}
    shifts: dict[str, list[float]] = {"rotated": [], "others": []}
    with np.errstate(over="ignore", invalid="ignore"):  # the run without a baseline diverges
        for seed in SEEDS:
            for name, rule in example.rules(seed).items():
                rewards, rotated, others = example.run(rule, seed, 320)
                final_rewards.setdefault(name, []).append(float(np.mean(rewards[-20:])))
                if name == "EH rule":
                    shifts["rotated"].append(float(np.mean(rotated)))
                    shifts["others"].append(float(np.mean(others)))
    assert sum(reward > 0.9 for reward in final_rewards["EH rule"]) >= MIN_WINS  # noqa: PLR2004
    # Lower is better for both: the negated reward, and the shift of the units the decoder reads right.
    _assert_lower(
        [-reward for reward in final_rewards["EH rule"]],
        [-reward for reward in final_rewards["without a reward baseline"]],
        margin=0.3,
    )
    _assert_lower(shifts["others"], shifts["rotated"], margin=5.0)


def test_delayed_response_needs_a_supralinear_eligibility() -> None:
    """Miconi: recurrent weights learn the delayed response with the cube, not with a linear eligibility."""
    example = _load_example("reward_modulated_delayed_response")
    final_errors: dict[str, list[float]] = {"cube": [], "linear": []}
    answered = []
    for seed in SEEDS:
        for name, power in (("cube", 3.0), ("linear", 1.0)):
            esn = example.build_model(seed, eligibility_power=power)
            errors = example.train(esn, 1500, seed)
            final_errors[name].append(float(np.mean(errors[-100:])))
            if name == "cube":
                answered.append(example.evaluate(esn)[0])
    _assert_lower(final_errors["cube"], final_errors["linear"], margin=0.2)
    assert sum(correct == 2 for correct in answered) >= MIN_WINS  # noqa: PLR2004
