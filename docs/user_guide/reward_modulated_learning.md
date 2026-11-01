# Learning from Rewards

Supervised readouts learn from targets. A **reward-modulated** model learns from a scalar
reward instead: it explores by perturbing its own activity, and correlates the perturbations with
the reward it receives. This suits goal-directed tasks where no target output is known, only how
good the behaviour was: control, decision making, and closed loops with an environment or a body.
The [theory page](../theory/reward_modulated_hebbian.md) explains the rule and why it works.

## The Closed Loop

A reward-modulated model is driven with two calls:

- `ESN.act(x)` steps the model through the input rows, adds exploration noise and records each
  step's **eligibility**, and returns the outputs.
- `ESN.reinforce(reward)` changes the weights in proportion to the eligibility and to how the
  reward compares with the reward expected so far.

```python
import numpy as np
from rclib import ESN, learning, readouts, reservoirs

rule = learning.hoerzer2014(5e-3, learning_rate_decay_steps=10_000.0)
esn = ESN()
esn.add_reservoir(reservoirs.RandomSparse(n_neurons=300, spectral_radius=1.2, leak_rate=0.1, seed=0))
esn.set_readout(readouts.RewardModulated(1, rule, include_bias=False))

output = np.zeros((1, 1))
for step in range(100_000):
    output = esn.act(output)  # the noisy output is fed back as the next input
    target = 0.5 * np.sin(2 * np.pi * step / 100)
    esn.reinforce(-float((output[0, 0] - target) ** 2))
```

The readout never sees the target: it only learns whether each step's error beat its recent
average. Afterwards `esn.predict_generative(np.empty((0, 1)), 500)` generates the sine on its own,
without noise.

## Building Blocks

**The rule.** `learning.RewardModulatedHebbian` holds every setting, grouped with
`learning.Exploration` (the noise) and `learning.Baseline` (the expected reward). It is a frozen
dataclass: build it with keywords and copy it with `dataclasses.replace`.

```python
rule = learning.RewardModulatedHebbian(
    learning_rate=0.05,
    exploration=learning.Exploration(distribution="uniform", amplitude=0.2, seed=1),
    post="high_pass",  # the potential minus its recent average estimates the noise
    update="online",  # update at every reward
    modulator="difference",  # R - R_avg
    baseline=learning.Baseline(kind="ema", decay=0.8),
)
```

**The presets** `learning.legenstein2010`, `learning.hoerzer2014`, `learning.miconi2017` and
`learning.burms2015` set the choices of four published variants and take the learning rate and
any setting to override. They are starting points: the learning rate and the exploration
amplitude depend on the task.

**Where it learns.**

- `readouts.RewardModulated(n_outputs, rule, include_bias=...)` is a linear readout whose output
  gets the exploration noise. Its weights start at zero at the first `act`;
  `esn.get_readout().setWeights(W)` sets them first.
- `reservoirs.RewardModulated(n_neurons, spectral_radius, ..., rule=rule)` is a random sparse
  reservoir whose recurrent weights learn. Its neurons integrate a potential with the leak rate
  and fire at its `tanh`, so a perturbation lingers for about `1 / leak_rate` steps.
  `plastic_presynaptic` and `plastic_postsynaptic` list the neurons whose synapses learn; the
  others keep their weights.

A model can hold reward-modulated reservoirs, a reward-modulated readout, or both: `reinforce`
updates all of them, each with its own baseline.

## Recipes

**A reward every step.** Call `act` with one row and `reinforce` after it, with an online rule
such as `legenstein2010` or `hoerzer2014`.

**One reward per trial.** Use a trial rule (`update="trial"`, as in `miconi2017` and
`burms2015`): act through the whole trial, then reinforce once. The eligibility sums the trial and
the update clears it. Pass the trial type as `key` so that each type has its own baseline:

```python
esn.reset_reservoirs()  # each trial from rest
rates = esn.act(trial_inputs)  # one call for the whole trial
esn.reinforce(-error, key=trial_type)
```

**Output feedback.** Append the output to the next input: the input weights drawn for those
columns act as feedback weights, uniform in `[-input_scaling, input_scaling]` times the factor
you scale the output by. With the output as the only input, `predict_generative` runs the
noise-free closed loop.

**Learning from observations.** `reservoirs.Nvar(num_lags=1, polynomial_order=1)` passes its
input through unchanged, so a reward-modulated readout learns directly from observations, for
example the sensors of a robot or a simulated body that plays the role of the reservoir.

**Outputs from the reservoir.** A model without a readout returns the reservoir's rates from
`act`, so designated neurons can be the outputs while the recurrent weights learn. Exclude them
from the plastic lists to keep their synapses fixed.

**Your own critic.** `reinforce(reward, baseline=b)` compares the reward with `b` instead of the
internal baseline, which is then left unchanged; `modulate(m)` applies a modulator directly, for
example a temporal-difference error.

**Supervised start, reward fine-tuning.** Fit a `Ridge` readout, read its weights, then switch to
a reward-modulated readout that starts from them:

```python
weights = esn.get_readout().getWeights()
esn.set_readout(readouts.RewardModulated(n_outputs, rule, include_bias=True))
esn.get_readout().setWeights(weights)
```

## Details

- `act(x, explore=False)` is a noise-free step that changes no learning state; with a readout it
  equals `predict_online`. `predict` and `predict_generative` are noise-free too.
- `reinforce` needs an exploring `act` since the previous update; otherwise it raises
  `RuntimeError` and changes nothing.
- The first reward of a key with an `"ema"` or `"window"` baseline becomes its own baseline, so it
  changes no weights; the update still counts toward `learning_rate_decay_steps`.
- `reset_reservoirs()` resets the activity, including the averages the high-pass signal uses, but
  keeps the eligibility. `clear_eligibility()` drops it, for example after an aborted trial.
- Updates apply to all reward-modulated components or to none: a non-finite reward raises
  `ValueError`, and a row or update that would overflow raises `RuntimeError`, in each case
  without changing any weights, eligibility or baselines.
- A reservoir object may appear only once in a model that acts.
- `fit` and `partial_fit` do not apply to a reward-modulated readout.
- The model file format stores the rule and the whole learning state, including the noise
  generator, so a model saved, pickled or copied in the middle of learning continues exactly. A
  model without a readout can be saved too.
- `get_readout().getLearning()` and `get_reservoir(i).getLearning()` give read-only access to the
  eligibility, baselines and update count.

## Examples

Each example in `examples/python/` distills one paper's main claim at a small scale; its outcome
is a property of that demonstration, not a reproduction of the paper.

| Example | After | Shows |
| :--- | :--- | :--- |
| `reward_modulated_pattern_generation.py` | Hoerzer et al. (2014) | A chaotic reservoir with output feedback learns to generate a sine from a reward alone; without exploration it learns nothing. Also in C++. |
| `reward_modulated_bci.py` | Legenstein et al. (2010) | A population adapts to a cursor decoder with half its directions rotated; the rotated units change their tuning more. |
| `reward_modulated_delayed_response.py` | Miconi (2017) | Recurrent weights learn to hold a cue through a delay from one reward per trial, with the cube eligibility but not a linear one. |
| `reward_modulated_body_control.py` | Burms et al. (2015) | Sensor feedback for a simulated mass-spring body learns from trial rewards; the decorrelated rule beats the plain one with redundant sensors. |

## C++

The same components are available in C++: `RewardModulatedHebbianRule`,
`RewardModulatedReadout`, `RewardModulatedReservoir`, and `Model::act`, `Model::reinforce`,
`Model::modulate` and `Model::clearEligibility`. See
`examples/cpp/reward_modulated_pattern_generation.cpp`.
