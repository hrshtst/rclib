# Reservoir Computing Overview

Reservoir Computing (RC) is a framework for training Recurrent Neural Networks (RNNs) where the recurrent part (the "reservoir") is fixed, and only the output weights are trained. This allows for extremely fast training compared to traditional Backpropagation Through Time (BPTT).

## Echo State Networks (ESN)

The standard Echo State Network (ESN) update equation is:

$$ \mathbf{x}(t+1) = (1-\alpha)\mathbf{x}(t) + \alpha \tanh(\mathbf{W}_{in}\mathbf{u}(t+1) + \mathbf{W}_{res}\mathbf{x}(t)) $$

Where:
*   $\mathbf{x}(t)$ is the reservoir state vector.
*   $\mathbf{u}(t)$ is the input vector.
*   $\mathbf{W}_{in}$ is the input weight matrix.
*   $\mathbf{W}_{res}$ is the reservoir weight matrix.
*   $\alpha$ is the leaking rate ($\alpha \in (0, 1]$).

The initial transient is usually discarded with a washout period before the
readout is fit. In `rclib`, `model.fit(..., washout_len=k)` advances the
reservoir over the full input sequence and trains only on rows `k:` of the
collected state matrix and target matrix.

`RandomSparse` optionally adds a fixed random bias vector inside the reservoir
update. Readouts also have their own `include_bias` option, which appends a
constant feature to the collected reservoir state before solving or updating
output weights.

## Next-Generation Reservoir Computing (NVAR)

NVAR replaces the random recurrent reservoir with a deterministic feature map of
time-delayed inputs. Let:

$$ \mathbf{z}(t) = [\mathbf{u}(t), \mathbf{u}(t-1), \ldots, \mathbf{u}(t-k+1)] $$

where $k$ is `num_lags`. For `polynomial_order=1`, the state is exactly this
linear delay embedding. For higher orders, `rclib` appends all monomials with
replacement for degrees `1..polynomial_order`, in deterministic lexicographic
index order. For example, if $\mathbf{z}=[z_0,z_1]$ and
`polynomial_order=2`, the feature vector is:

$$ [z_0, z_1, z_0^2, z_0z_1, z_1^2] $$

Constant/bias features are not part of the NVAR reservoir state; use the
readout's `include_bias` option when an intercept term is needed.
