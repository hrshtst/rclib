# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Learn to generate a sine wave from a reward signal alone, after Hoerzer et al. (2014).

A chaotic reservoir receives its own output as input (output feedback), and a linear readout
explores by adding noise to that output. The readout never sees the target: each step it only
learns whether the squared error beat its recent average (a Heaviside modulator), and it
correlates that with its own exploration noise. After training, the model generates the sine on
its own, without noise, through ``predict_generative``. A paired run without exploration noise
learns nothing, since the noise is what the rule correlates with the reward.

This is a small demonstration of the idea, not a reproduction of the paper's results; the
outcome is a property of this setup (its reservoir size, spectral radius and learning rate).
"""

from __future__ import annotations

import sys

import numpy as np
from rclib import ESN, learning, readouts, reservoirs

PERIOD = 100  # steps per cycle of the target
AMPLITUDE = 0.5


def target(steps: np.ndarray) -> np.ndarray:
    """The sine wave the readout should generate."""
    return AMPLITUDE * np.sin(2 * np.pi * steps / PERIOD)


def build_model(seed: int, *, exploration_amplitude: float = 0.2) -> ESN:
    """A chaotic reservoir fed back its noisy output, with a reward-modulated readout."""
    rule = learning.hoerzer2014(
        5e-3,
        exploration=learning.Exploration(amplitude=exploration_amplitude, seed=seed),
        learning_rate_decay_steps=10_000.0,
    )
    esn = ESN()
    # The input is the fed-back output, so input_scaling sets the feedback weights, U[-1, 1].
    esn.add_reservoir(
        reservoirs.RandomSparse(n_neurons=300, spectral_radius=1.2, sparsity=0.1, leak_rate=0.1, seed=seed)
    )
    esn.set_readout(readouts.RewardModulated(1, rule, include_bias=False))
    return esn


def train(esn: ESN, steps: int) -> np.ndarray:
    """Run the closed loop for ``steps`` steps, rewarding -error^2; return the squared errors."""
    output = np.zeros((1, 1))
    errors = np.empty(steps)
    for step in range(steps):
        output = esn.act(output)  # the noisy output is fed back as the next input
        error = float(output[0, 0] - target(np.array(step)))
        esn.reinforce(-(error**2))
        errors[step] = error**2
    return errors


def generate(esn: ESN, steps: int) -> np.ndarray:
    """Continue the pattern without noise, feeding each output back as the next input."""
    return esn.predict_generative(np.empty((0, 1)), steps)[:, 0]


def normalized_error(generated: np.ndarray, start: int) -> float:
    """Mean squared error of a generated continuation against the target, over the target's variance."""
    expected = target(np.arange(start, start + len(generated)))
    return float(np.mean((generated - expected) ** 2) / np.var(expected))


def main() -> None:
    """Train with and without exploration, then generate the pattern without noise."""
    train_steps, test_steps, seed = 100_000, 500, 0
    results = {}
    for name, amplitude in (("exploring", 0.2), ("no exploration", 0.0)):
        esn = build_model(seed, exploration_amplitude=amplitude)
        errors = train(esn, train_steps)
        generated = generate(esn, test_steps)
        score = normalized_error(generated, train_steps)
        results[name] = (errors, generated, score)
        print(f"{name}: normalized error of the generated pattern {score:.3f}")
    plot(results, train_steps)


def plot(results: dict[str, tuple[np.ndarray, np.ndarray, float]], train_steps: int) -> None:
    """Plot the training error and the generated patterns."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot.")
        return
    fig, (ax_train, ax_test) = plt.subplots(2, 1, figsize=(10, 7))
    window = 1000
    for name, (errors, generated, score) in results.items():
        smoothed = np.convolve(errors, np.ones(window) / window, mode="valid")
        ax_train.plot(np.arange(len(smoothed)) + window, smoothed, label=name)
        ax_test.plot(generated, label=f"{name} (normalized error {score:.3f})")
    test_steps = len(next(iter(results.values()))[1])
    ax_test.plot(target(np.arange(train_steps, train_steps + test_steps)), "k--", label="target")
    ax_train.set(title="Training error, noisy output (moving average)", xlabel="step", ylabel="squared error")
    ax_train.set_yscale("log")
    ax_test.set(title="Generated without noise after training", xlabel="step after training", ylabel="output")
    for axis in (ax_train, ax_test):
        axis.legend()
        axis.grid(visible=True)
    fig.tight_layout()
    output_file = "reward_modulated_pattern_generation.png"
    fig.savefig(output_file)
    print(f"Plot saved to {output_file}")
    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
