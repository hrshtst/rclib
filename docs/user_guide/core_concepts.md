# Core Concepts

## Configuring Reservoirs

The reservoir is the dynamical core of the ESN. `rclib` provides `RandomSparse` for standard ESNs.

```python
res = reservoirs.RandomSparse(
    n_neurons=1000,  # Size of the reservoir
    spectral_radius=0.9,  # Scaling of spectral radius
    sparsity=0.1,  # Density of connections
    leak_rate=1.0,  # 1.0 = full update, < 1.0 = leaky integrator
    input_scaling=1.0,  # Scaling of input weights
    include_bias=False,  # Add bias neuron to reservoir
    seed=42,  # Random seed for reproducibility
    spectral_radius_method="power_iteration",  # or "dense": exact, O(n^3) time, small reservoirs only
)
```

`spectral_radius_method` chooses how the spectral radius of the random weight
matrix is found before the matrix is scaled to `spectral_radius`. The default
`"power_iteration"` estimates it. In measurements the scaled radius was
usually within about 0.01% of the request and at worst 0.75% off, but these are
observations, not bounds. `"dense"` is exact but costs $O(n^3)$ time and
$O(n^2)$ memory in `n_neurons`, so use it for small reservoirs or when the radius
must be exact. See
[Spectral Radius Scaling](advanced_usage.md#spectral-radius-scaling).

## Configuring Readouts

The readout maps the high-dimensional reservoir state to the target output.

*   **Ridge Regression (`readouts.Ridge`)**: The standard offline training method. Fast and stable.
*   **Recursive Least Squares (`readouts.Rls`)**: For online, adaptive learning.
*   **Least Mean Squares (`readouts.Lms`)**: A simpler gradient-based online method.

Ridge readouts regularize all fitted weights, including the appended constant
feature when `include_bias=True`.

## Building the Model

The `ESN` class acts as a container.

```python
model = ESN(connection_type="serial")  # "serial" or "parallel"
model.add_reservoir(res1)
# For deep ESNs:
# model.add_reservoir(res2)
model.set_readout(readout)
```

## Training

`model.fit(x, y, washout_len=k)` resets the reservoirs, runs them through the
whole input sequence and fits the readout on all but the first `k` states. For
several independent sequences, such as episodes, use
`model.fit_sequences([x1, x2], [y1, y2], washout_len=k)`: each sequence starts
from reset reservoirs and loses its own washout before one readout is fitted on
all of them. See
[Training on Several Sequences](advanced_usage.md#training-on-several-sequences).
