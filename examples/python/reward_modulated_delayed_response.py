# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Learn a working-memory task in the recurrent weights from a delayed reward, after Miconi (2017).

A reward-modulated reservoir sees one of two cues, then a delay without input, then must answer
with neuron 0: positive after cue A, negative after cue B. Its recurrent weights learn from a
single reward at the end of each trial, minus the mean absolute error of neuron 0 over the
response period, compared with a running average of earlier rewards for the same cue. During the
trial, sparse random kicks perturb the neurons' potentials, and the eligibility sums the cube of
each synapse's Hebbian product: the large products of the kicks dominate the small ones of the
network's relaxation. With the cube the network learns to hold the cue through the delay; with a
linear eligibility the same rule stays at chance, as the paper reports.

The model has no readout: ``ESN.act`` returns the rates, and neuron 0 is the output. Its synapses
learn like all others.

This is a small demonstration of the idea, not a reproduction of the paper's results; the
outcome is a property of this setup (its network size, timing and learning rate).
"""

from __future__ import annotations

import sys

import numpy as np
from rclib import ESN, learning, reservoirs

CUE, DELAY, RESPONSE = 50, 50, 50  # steps of the trial's phases
TARGETS = (0.5, -0.5)  # neuron 0 during the response, after cue A and after cue B


def trial_input(cue: int) -> np.ndarray:
    """One trial's inputs: cue A or B on its own channel, then nothing during the delay and response."""
    inputs = np.zeros((CUE + DELAY + RESPONSE, 2))
    inputs[:CUE, cue] = 1.0
    return inputs


def build_model(seed: int, *, eligibility_power: float = 3.0) -> ESN:
    """A chaotic potential-leak reservoir whose recurrent weights learn by trials, without a readout."""
    rule = learning.miconi2017(
        0.5,
        eligibility_power=eligibility_power,
        max_update=1e-3,
        exploration=learning.Exploration(amplitude=0.5, probability=0.003, seed=seed),
    )
    esn = ESN()
    esn.add_reservoir(
        reservoirs.RewardModulated(100, 1.5, sparsity=1.0, leak_rate=0.1, rule=rule, include_bias=True, seed=seed)
    )
    return esn


def train(esn: ESN, trials: int, seed: int) -> np.ndarray:
    """Run `trials` trials with random cues, one reward each; return the error of each trial."""
    rng = np.random.default_rng(seed)
    errors = np.empty(trials)
    for trial in range(trials):
        cue = int(rng.integers(2))
        esn.reset_reservoirs()
        rates = esn.act(trial_input(cue))
        errors[trial] = np.mean(np.abs(rates[-RESPONSE:, 0] - TARGETS[cue]))
        # One reward at the end of the trial; the baseline is kept per cue.
        esn.reinforce(-errors[trial], key=cue)
    return errors


def evaluate(esn: ESN) -> tuple[int, list[np.ndarray]]:
    """Noise-free trials of both cues: how many answer with the right sign, and neuron 0's traces."""
    correct, traces = 0, []
    for cue in (0, 1):
        esn.reset_reservoirs()
        trace = esn.act(trial_input(cue), explore=False)[:, 0]
        correct += int(np.sign(trace[-RESPONSE:].mean()) == np.sign(TARGETS[cue]))
        traces.append(trace)
    return correct, traces


def main() -> None:
    """Train with the cube and with a linear eligibility, then test both cues without noise."""
    seed, trials = 0, 3000
    results = {}
    for name, power in (("cube", 3.0), ("linear", 1.0)):
        esn = build_model(seed, eligibility_power=power)
        errors = train(esn, trials, seed)
        correct, traces = evaluate(esn)
        results[name] = (errors, traces)
        print(
            f"{name} eligibility: error {errors[:100].mean():.3f} over the first 100 trials, "
            f"{errors[-100:].mean():.3f} over the last 100; {correct} of 2 cues answered correctly"
        )
    plot(results)


def plot(results: dict[str, tuple[np.ndarray, list[np.ndarray]]]) -> None:
    """Plot the learning curves and neuron 0's traces after training."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot.")
        return
    fig, (ax_error, ax_trace) = plt.subplots(1, 2, figsize=(12, 4.5))
    window = 50
    for name, (errors, _) in results.items():
        smoothed = np.convolve(errors, np.ones(window) / window, mode="valid")
        ax_error.plot(np.arange(len(smoothed)) + window, smoothed, label=f"{name} eligibility")
    ax_error.set(title="Error at the end of each trial (moving average)", xlabel="trial", ylabel="mean |error|")
    ax_error.legend()
    ax_error.grid(visible=True)

    steps = np.arange(CUE + DELAY + RESPONSE)
    for name, style in (("cube", "-"), ("linear", ":")):
        for cue, color in ((0, "tab:blue"), (1, "tab:orange")):
            ax_trace.plot(steps, results[name][1][cue], style, color=color, label=f"{name}, cue {'AB'[cue]}")
    for cue, color in ((0, "tab:blue"), (1, "tab:orange")):
        ax_trace.hlines(TARGETS[cue], CUE + DELAY, CUE + DELAY + RESPONSE, colors=color, linestyles="--")
    ax_trace.axvspan(0, CUE, color="gray", alpha=0.15, label="cue")
    ax_trace.axvspan(CUE + DELAY, CUE + DELAY + RESPONSE, color="green", alpha=0.1, label="response")
    ax_trace.set(title="Neuron 0 after training, without noise", xlabel="step of the trial", ylabel="rate")
    ax_trace.legend(fontsize="small")
    ax_trace.grid(visible=True)
    fig.tight_layout()
    output_file = "reward_modulated_delayed_response.png"
    fig.savefig(output_file)
    print(f"Plot saved to {output_file}")
    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
