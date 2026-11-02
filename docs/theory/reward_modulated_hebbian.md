# Reward-Modulated Hebbian Learning

The readouts of the other theory pages learn from targets: they are told the output they should
have produced. Reward-modulated Hebbian learning needs much less. The model only receives a
scalar reward $R$ that says how good its recent behaviour was, and learns by **exploring**:
it perturbs its own activity and correlates the perturbations with the reward. No target, no
error per output and no gradient is given, so the same rule can learn goal-directed behaviour in a
closed loop with an environment, a body or a user.

`rclib` implements one composable rule, `learning.RewardModulatedHebbian`, that covers the
variants of four papers (see [References](#references)), and two places where it can learn: the
readout (`readouts.RewardModulated`) and the recurrent weights of a reservoir
(`reservoirs.RewardModulated`).

## The Rule

As everywhere in `rclib`, activities are row vectors and a weight matrix $\mathbf{W}$ maps a
pre-synaptic row $\mathbf{p}$ to the post-synaptic **potential** $\mathbf{v} = \mathbf{p}\mathbf{W}$,
so $W_{ij}$ is the weight from unit $i$ to unit $j$. Here $\mathbf{p}$ is the activity on the input
side of the weights that learn, not necessarily the model's input. For a reward-modulated readout
it is the feature row the readout receives: the reservoir state, with a trailing 1 when the readout
has a bias. For a reward-modulated reservoir it is the previous rate $\mathbf{r}(t-1)$ of its own
neurons, because only its recurrent weights learn; its input and bias add fixed terms to the
potential (see [Where the Rule Learns](#where-the-rule-learns)). The model's input is $\mathbf{p}$
only when an identity NVAR passes it straight to a reward-modulated readout.

While exploring, every post-synaptic unit adds noise $\boldsymbol{\xi}$ to its potential. Each
exploring step computes a post-synaptic signal $\mathbf{q}$ and adds the Hebbian product of the two
sides to an **eligibility** $\mathbf{E}$:

$$ \mathbf{E} \leftarrow \gamma \mathbf{E} + S(\mathbf{p})^{T} S(\mathbf{q}), \qquad S(y) = \operatorname{sign}(y)\,|y|^{k} $$

with $\gamma$ the `trace_decay` of online updates ($\gamma = 1$ within a trial, which clears
$\mathbf{E}$ at its update) and $k$ the `eligibility_power`. Since $S(ab) = S(a)S(b)$, the term is
an outer product. The post-synaptic signal is one of

- the noise itself, $\mathbf{q} = \boldsymbol{\xi}$ (`post="noise"`), when it is known;
- the potential minus its average before this step, $\mathbf{q} = \mathbf{v} - \bar{\mathbf{v}}$,
  with $\bar{\mathbf{v}} \leftarrow d\,\bar{\mathbf{v}} + (1 - d)\,\mathbf{v}$ (`post="high_pass"`,
  $d$ the `post_decay`; $d = 0$ gives the first difference of the potential);
- the potential itself (`post="potential"`).

A reward then changes the weights in proportion to the eligibility and a **modulator** $m$:

$$ \Delta \mathbf{W} = \eta_n\, m\, \mathbf{E} \qquad \text{or, decorrelated,} \qquad \Delta \mathbf{W} = \eta_n\, m\, (\mathbf{C} + \lambda \mathbf{I})^{-1} \mathbf{E} $$

where $\mathbf{C}$ sums $\mathbf{p}^{T}\mathbf{p}$ over the trial, $\lambda$ is the `decorrelation`,
and $\eta_n = \eta / (1 + n/T)$ decays with the number $n$ of updates when
`learning_rate_decay_steps` $T$ is set. The modulator compares the reward with a **baseline**
$\bar{R}$, kept per integer key such as a trial type, before the reward is added to it:

$$ m = R - \bar{R} \quad (\texttt{"difference"}) \qquad \text{or} \qquad m = \begin{cases} 1 & R > \bar{R} \\ 0 & \text{otherwise} \end{cases} \quad (\texttt{"heaviside"}) $$

The baseline is an exponential average of the key's rewards, the mean of the last few, zero, a
value the caller passes (any critic), or bypassed entirely by giving $m$ directly with
`ESN.modulate`. The first reward of a key with an internal baseline becomes its own baseline, so
its modulator is zero.

## Why It Works

Write the reward as a function of the potentials and suppose the noise is small, zero-mean and
independent across units and steps, with variance $\sigma^2$. Then, to first order,
$R - \bar{R} \approx \sum_j \frac{\partial R}{\partial v_j} \xi_j$, and the expected change of a
weight under $\mathbf{q} = \boldsymbol{\xi}$, $k = 1$ is

$$ \mathbb{E}[\Delta W_{ij}] = \eta\, \mathbb{E}\!\left[p_i\, \xi_j\, (R - \bar{R})\right] \approx \eta\, \sigma^{2}\, p_i \frac{\partial R}{\partial v_j} = \eta\, \sigma^{2} \frac{\partial R}{\partial W_{ij}} . $$

The rule climbs the gradient of the reward without computing it: this is node perturbation.
The other choices make it work with less information or in harder conditions:

- **The baseline** removes the part of the reward that does not depend on the noise. Without it
  the updates have a large bias and variance, which can make learning collapse.
- **The high-pass signal** estimates noise the synapse cannot observe: when the inputs change
  slowly, the potential minus its recent average is mostly the noise. This is the exploratory
  Hebbian rule, local to the synapse.
- **The supralinear eligibility** ($k = 3$) helps when the reward comes only at the end of a trial.
  A kick to a recurrent network's potential is followed by a long, small relaxation that adds
  unrelated terms to the eligibility; the cube keeps the few large products of the kicks and
  suppresses the many small ones.
- **The Heaviside modulator** learns only from steps better than expected, a coarser but robust
  signal.
- **Decorrelation** preconditions the update with the inverse covariance of the pre-synaptic
  activity, which matters when inputs are redundant, correlated or on different scales. Its step
  is normalized by the trial's covariance, so its learning rates are not comparable with the plain
  rule's.

## Published Variants

The presets in `rclib.learning` set the choices that define each variant; the learning rate and
the exploration amplitude depend on the task, and every setting can be overridden. They are
starting points, not reproductions of the papers.

| Preset | Learns | Exploration | Post signal | Eligibility | Modulator, baseline | Update |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `legenstein2010` | the layer feeding the explored units | uniform, every unit and step | high-pass, $d = 0.8$ | linear | difference, EMA 0.8 | every rewarded step |
| `hoerzer2014` | a readout whose noisy output is fed back | uniform, every step | first difference | linear | Heaviside, EMA 0.8 | every step, decaying rate |
| `miconi2017` | recurrent weights | sparse kicks (probability 0.003, amplitude 0.5) | high-pass, $d = 0.05$ | cube | difference, EMA 0.33 per trial type | once per trial, clipped |
| `burms2015` | recurrent weights or sensor feedback | Gaussian 0.05, every step | the known noise | linear | difference, mean of the last 50 per input | once per trial, decorrelated |

## Where the Rule Learns

**A reward-modulated readout** adds the noise to its outputs, $\mathbf{z} = \mathbf{x}\mathbf{W} + \boldsymbol{\xi}$,
where $\mathbf{x}$ holds the features (and a trailing 1 with a bias). To feed the output back into
the reservoir, as Hoerzer et al. do, append it to the next input: since the input weights are drawn
uniformly from $[-s, s]$ with $s$ the `input_scaling`, the appended columns act as feedback weights
with the same distribution. With an identity NVAR (`Nvar(num_lags=1, polynomial_order=1)`) the
readout learns directly from observations, for example from the sensors of a body that plays the
role of the reservoir.

**A reward-modulated reservoir** learns its recurrent weights. Its neurons integrate a potential,

$$ \mathbf{v}(t) = (1 - \alpha)\,\mathbf{v}(t-1) + \alpha\left(\mathbf{u}(t)\mathbf{W}_{in} + \mathbf{b} + \mathbf{r}(t-1)\mathbf{W}\right) + \boldsymbol{\xi}(t), \qquad \mathbf{r}(t) = \tanh \mathbf{v}(t), $$

with the noise outside the leak, so that a kick stays in the dynamics for about $1/\alpha$ steps.
The pre-synaptic activity is the previous rate $\mathbf{r}(t-1)$ and the post-synaptic one the
potential. Learning changes only existing synapses, and optional lists of plastic pre- and
post-synaptic neurons keep the rest fixed, for example the synapses of designated output neurons.
With $\alpha = 1$ and no noise the reservoir equals a `RandomSparse` reservoir with the same
weights.

## References

- R. Legenstein, S. M. Chase, A. B. Schwartz and W. Maass. A reward-modulated Hebbian learning
  rule can explain experimentally observed network reorganization in a brain control task.
  *The Journal of Neuroscience*, 30(25):8400–8410, 2010.
- G. M. Hoerzer, R. Legenstein and W. Maass. Emergence of complex computational structures from
  chaotic neural networks through reward-modulated Hebbian learning. *Cerebral Cortex*,
  24(3):677–690, 2014.
- T. Miconi. Biologically plausible learning in recurrent neural networks reproduces neural
  dynamics observed during cognitive tasks. *eLife*, 6:e20899, 2017.
- J. Burms, K. Caluwaerts and J. Dambre. Reward-modulated Hebbian plasticity as leverage for
  partially embodied control in compliant robotics. *Frontiers in Neurorobotics*, 9:9, 2015.
