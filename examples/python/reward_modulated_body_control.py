# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Learn sensor feedback for a simulated body from trial rewards, after Burms et al. (2015).

The body is a chain of six masses joined by stiff nonlinear springs and dampers, anchored at one
end. Two actuators push on two of the masses, and the free end, the tip, should follow a sine.
A crude feed-forward drive, the static force that would hold the tip at the target if the chain
were linear and slow, misses: the chain's dynamics amplify it and its springs stiffen. Linear
feedback from the body's sensors to the actuators corrects it:

    actuators = feedforward(t) + sensors(t) @ W + noise

The body plays the reservoir, as in physical reservoir computing: an identity NVAR passes the
sensor readings straight to a reward-modulated readout, whose weights W learn from one reward per
period of the sine, minus the tip's tracking error beyond a 1 cm dead zone. The sensors are
redundant and on very different scales (positions, velocities, scaled spring extensions and their
cubes, and the target itself), which slows the plain trial rule; the decorrelated rule, which
solves with the sensors' covariance over the trial, learns faster and further. Both use the
learning rate that worked best for them in a search over the same number of values.

This is a small demonstration of the idea, not a reproduction of the paper's results; the
outcome is a property of this setup. The chain is a toy defined in this file, not part of rclib.
"""

from __future__ import annotations

import sys

import numpy as np
from rclib import ESN, learning, readouts, reservoirs

N_MASSES = 6
STIFFNESS, CUBIC_STIFFNESS, DAMPING = 100.0, 5000.0, 1.0
ACTUATED = (1, 3)  # the masses the two actuators push
DT, SUBSTEPS = 0.05, 4
PERIOD, WARMUP = 100, 20  # steps per trial (one period of the target) and steps not scored
AMPLITUDE, DEAD_ZONE = 0.3, 0.01


def target(step: float) -> float:
    """The tip's target position."""
    return AMPLITUDE * np.sin(2 * np.pi * step / PERIOD)


class Chain:
    """Masses in a line, joined to each other and to a wall by springs F = k e + k3 e^3 and dampers."""

    def __init__(self) -> None:
        """Start at rest."""
        self.position = np.zeros(N_MASSES)
        self.velocity = np.zeros(N_MASSES)

    def step(self, forces: np.ndarray) -> None:
        """Advance one step with the actuator forces, by semi-implicit Euler substeps."""
        h = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            extension = np.diff(self.position, prepend=0.0)
            tension = (
                STIFFNESS * extension + CUBIC_STIFFNESS * extension**3 + DAMPING * np.diff(self.velocity, prepend=0.0)
            )
            acceleration = np.append(tension[1:], 0.0) - tension  # unit masses
            acceleration[list(ACTUATED)] += forces
            self.velocity += h * acceleration
            self.position += h * self.velocity

    def sensors(self, step: int) -> np.ndarray:
        """Redundant readings on very different scales, and the target with its rate of change."""
        extension = np.diff(self.position, prepend=0.0)
        rate = AMPLITUDE * 2 * np.pi / PERIOD * np.cos(2 * np.pi * step / PERIOD)
        return np.concatenate(
            (self.position, self.velocity, 10 * extension, 100 * extension**3, [target(step), 10 * rate])
        )


def feedforward(step: int) -> np.ndarray:
    """The static force at the outer actuator that would hold a linear chain's tip at the target."""
    return np.array([0.0, target(step) * STIFFNESS / (ACTUATED[1] + 1)])


def run_trial(esn: ESN, *, explore: bool = True) -> tuple[float, np.ndarray]:
    """One period from rest; return the mean tracking error beyond the dead zone and the tip's path."""
    body = Chain()
    tip = np.empty(PERIOD)
    errors = []
    esn.reset_reservoirs()
    for step in range(PERIOD):
        feedback = esn.act(body.sensors(step).reshape(1, -1), explore=explore)[0]
        body.step(feedforward(step) + feedback)
        tip[step] = body.position[-1]
        if step >= WARMUP:
            errors.append(max(abs(tip[step] - target(step + 1)) - DEAD_ZONE, 0.0))
    return float(np.mean(errors)), tip


def build_model(rule: learning.RewardModulatedHebbian) -> ESN:
    """The body's sensors, through an identity NVAR, to a reward-modulated readout for two actuators."""
    esn = ESN()
    esn.add_reservoir(reservoirs.Nvar(num_lags=1, polynomial_order=1))  # passes the sensors through
    esn.set_readout(readouts.RewardModulated(len(ACTUATED), rule, include_bias=True))
    return esn


def rule(learning_rate: float, seed: int, *, decorrelated: bool) -> learning.RewardModulatedHebbian:
    """The trial rule of Burms et al. with Gaussian exploration, decorrelated or plain."""
    return learning.burms2015(
        learning_rate,
        exploration=learning.Exploration(distribution="gaussian", amplitude=0.5, seed=seed),
        decorrelation=1.0 if decorrelated else None,
    )


def train(rule: learning.RewardModulatedHebbian, trials: int) -> tuple[np.ndarray, float, ESN]:
    """Train for `trials` trials, one reward each; return the errors, the final noise-free error and the model.

    A learning rate too large for the rule makes the body diverge, which is reported as an
    infinite final error.
    """
    esn = build_model(rule)
    errors = np.full(trials, np.inf)
    # A diverging body overflows on its way to non-finite sensors, which act then rejects.
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            for trial in range(trials):
                errors[trial], _ = run_trial(esn)
                esn.reinforce(-errors[trial])
            final, _ = run_trial(esn, explore=False)
        except (ValueError, RuntimeError):  # the body diverged
            final = np.inf
    return errors, final, esn


# The learning rates that worked best in a search over 0.25 to 4 (plain) and 25 to 400
# (decorrelated), five values each. Their scales differ because the decorrelated update solves
# with the covariance summed over the trial.
LEARNING_RATES = {"plain": 1.0, "decorrelated": 100.0}


def main() -> None:
    """Train the plain and the decorrelated rule, and compare the tip's path before and after."""
    seed, trials = 0, 400
    initial_error, initial_tip = run_trial(build_model(rule(1.0, seed, decorrelated=False)), explore=False)
    print(f"feed-forward only: tracking error {initial_error:.4f}")
    results = {}
    for name in ("plain", "decorrelated"):
        errors, final, esn = train(rule(LEARNING_RATES[name], seed, decorrelated=name == "decorrelated"), trials)
        _, tip = run_trial(esn, explore=False)
        results[name] = (errors, tip)
        print(f"{name} rule: tracking error {final:.4f} after {trials} trials")
    plot(results, initial_tip)


def plot(results: dict[str, tuple[np.ndarray, np.ndarray]], initial_tip: np.ndarray) -> None:
    """Plot the learning curves and the tip's path."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot.")
        return
    fig, (ax_error, ax_tip) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (errors, _) in results.items():
        ax_error.plot(np.arange(1, len(errors) + 1), errors, label=f"{name} rule")
    ax_error.set(title="Tracking error beyond the dead zone", xlabel="trial", ylabel="mean error")
    ax_error.set_yscale("log")
    ax_error.legend()
    ax_error.grid(visible=True)

    steps = np.arange(1, PERIOD + 1)
    ax_tip.plot(steps, [target(float(step)) for step in steps], "k--", label="target")
    ax_tip.plot(steps, initial_tip, color="gray", label="feed-forward only")
    for name, (_, tip) in results.items():
        ax_tip.plot(steps, tip, label=f"{name} rule, trained")
    ax_tip.axvspan(0, WARMUP, color="gray", alpha=0.1, label="not scored")
    ax_tip.set(title="Tip position over one period, without noise", xlabel="step", ylabel="position")
    ax_tip.legend(fontsize="small")
    ax_tip.grid(visible=True)
    fig.tight_layout()
    output_file = "reward_modulated_body_control.png"
    fig.savefig(output_file)
    print(f"Plot saved to {output_file}")
    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
