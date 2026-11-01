# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Adapt to a perturbed cursor decoder with the exploratory Hebbian rule, after Legenstein et al. (2010).

A population of 40 units drives a 2-D cursor toward targets, as in a brain-computer interface.
The units receive 16 cosine-tuned inputs that encode the desired direction, through an identity
NVAR, so the plastic readout is the layer from the inputs to the units. A population vector
decodes the cursor velocity, but the decoder believes the wrong preferred direction for half of
the units: theirs are rotated by 90 degrees. Every step the units explore with noise, and the
only feedback is a scalar reward, the cosine between the cursor's velocity and the direction to
the target. The exploratory Hebbian (EH) rule correlates each unit's noisy activity with the
reward and recovers the cursor control. The rotated units also shift their tuning more than the
others, the credit-assignment effect the paper reports: the global reward reaches the units that
caused the error. A paired run without a reward baseline, whose modulator is then the reward
itself, learns at first but collapses after a few dozen targets.

This is a small demonstration of the idea, not a reproduction of the paper's results; the
outcomes are properties of this setup (its sizes, noise amplitude and learning rate).
"""

from __future__ import annotations

import sys

import numpy as np
from rclib import ESN, learning, readouts, reservoirs

N_INPUTS = 16
N_UNITS = 40
N_TARGETS = 8
INPUT_ANGLES = np.linspace(0, 2 * np.pi, N_INPUTS, endpoint=False)
TARGET_ANGLES = np.linspace(0, 2 * np.pi, N_TARGETS, endpoint=False)
SPEED = 0.05  # cursor displacement per step for a unit velocity
HIT_RADIUS = 0.05
MAX_STEPS = 200  # per target


def encode(direction: np.ndarray) -> np.ndarray:
    """The inputs for a desired direction: one cosine-tuned input per preferred angle."""
    angle = np.arctan2(direction[1], direction[0])
    return np.cos(angle - INPUT_ANGLES).reshape(1, -1)


def tuning(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preferred directions and modulation depths of the units.

    A unit's noise-free response to the direction theta is the sum over inputs j of
    w_j cos(theta - phi_j) = depth cos(theta - preferred).
    """
    cosine = np.cos(INPUT_ANGLES) @ weights
    sine = np.sin(INPUT_ANGLES) @ weights
    return np.arctan2(sine, cosine), np.hypot(cosine, sine)


def run(rule: learning.RewardModulatedHebbian, seed: int, n_targets: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reach `n_targets` random targets with half of the decoding directions rotated by 90 degrees.

    Returns the mean reward of each target and the tuning shift, in degrees, of the rotated and
    of the other units.
    """
    rng = np.random.default_rng(seed)
    initial_weights = rng.uniform(-0.5, 0.5, (N_INPUTS, N_UNITS))
    esn = ESN()
    esn.add_reservoir(reservoirs.Nvar(num_lags=1, polynomial_order=1))  # passes the inputs through
    esn.set_readout(readouts.RewardModulated(N_UNITS, rule, include_bias=False))
    esn.get_readout().setWeights(initial_weights)

    preferred, depth = tuning(initial_weights)
    rotated = np.zeros(N_UNITS, dtype=bool)
    rotated[rng.permutation(N_UNITS)[: N_UNITS // 2]] = True
    decoded = preferred + np.where(rotated, np.pi / 2, 0.0)
    decoding = np.stack([np.cos(decoded), np.sin(decoded)], axis=1)

    rewards = []
    for _ in range(n_targets):
        target_rewards = []
        angle = TARGET_ANGLES[rng.integers(N_TARGETS)]
        target = np.array([np.cos(angle), np.sin(angle)])
        cursor = np.zeros(2)
        for _ in range(MAX_STEPS):
            to_target = target - cursor
            distance = float(np.linalg.norm(to_target))
            if distance < HIT_RADIUS:
                break
            desired = to_target / distance
            activity = esn.act(encode(desired))[0]
            velocity = (2 / N_UNITS) * (activity / depth) @ decoding  # the population vector
            cursor += SPEED * velocity
            reward = float(velocity @ desired / (np.linalg.norm(velocity) + 1e-12))
            if not np.isfinite(reward):  # a diverging rule
                return np.array(rewards), np.full(1, np.nan), np.full(1, np.nan)
            esn.reinforce(reward)
            target_rewards.append(reward)
        rewards.append(np.mean(target_rewards))

    final_preferred, _ = tuning(esn.get_readout().getWeights())
    shift = np.degrees(np.abs(np.angle(np.exp(1j * (final_preferred - preferred)))))
    return np.array(rewards), shift[rotated], shift[~rotated]


def rules(seed: int) -> dict[str, learning.RewardModulatedHebbian]:
    """The EH rule, and the same rule without a reward baseline."""
    exploration = learning.Exploration(amplitude=1.0, seed=seed)
    return {
        "EH rule": learning.legenstein2010(0.005, exploration=exploration),
        "without a reward baseline": learning.legenstein2010(
            0.005, exploration=exploration, baseline=learning.Baseline(kind="none")
        ),
    }


def main() -> None:
    """Adapt to the perturbed decoder, with and without a reward baseline."""
    seed, n_targets = 0, 320
    results = {}
    with np.errstate(over="ignore", invalid="ignore"):  # the run without a baseline diverges
        for name, rule in rules(seed).items():
            results[name] = run(rule, seed, n_targets)
    for name, (rewards, rotated, others) in results.items():
        print(
            f"{name}: mean reward {rewards[:20].mean():.3f} over the first 20 targets, "
            f"{rewards[-20:].mean():.3f} over the last 20; "
            f"tuning shift {np.nanmean(rotated):.1f} deg (rotated units) vs {np.nanmean(others):.1f} deg (others)"
        )
    plot(results)


def plot(results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    """Plot the reward over time and the tuning shifts of the EH rule."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot.")
        return
    fig, (ax_reward, ax_shift) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (rewards, _, _) in results.items():
        ax_reward.plot(np.arange(1, len(rewards) + 1), rewards, label=name)
    ax_reward.set(title="Reward: cosine of cursor velocity and target direction", xlabel="target", ylabel="mean reward")
    ax_reward.set_ylim(-0.2, 1.05)
    ax_reward.legend()
    ax_reward.grid(visible=True)

    _, rotated, others = results["EH rule"]
    rng = np.random.default_rng(0)
    for position, shifts in enumerate((rotated, others)):
        ax_shift.bar(position, shifts.mean(), color="lightgray", edgecolor="black")
        ax_shift.scatter(position + rng.uniform(-0.2, 0.2, len(shifts)), shifts, s=12, color="tab:blue")
    ax_shift.set_xticks([0, 1], ["rotated units", "other units"])
    ax_shift.set(title="Tuning shift with the EH rule (bars: means)", ylabel="shift of preferred direction (deg)")
    ax_shift.grid(visible=True, axis="y")
    fig.tight_layout()
    output_file = "reward_modulated_bci.png"
    fig.savefig(output_file)
    print(f"Plot saved to {output_file}")
    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
