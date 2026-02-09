# Core Concepts

## Configuring Reservoirs

The reservoir is the dynamical core of the ESN. `rclib` provides `RandomSparse` for standard ESNs.

```python
res = reservoirs.RandomSparse(
    n_neurons=1000,      # Size of the reservoir
    spectral_radius=0.9, # Scaling of spectral radius
    sparsity=0.1,        # Density of connections
    leak_rate=1.0,       # 1.0 = full update, < 1.0 = leaky integrator
    input_scaling=1.0,   # Scaling of input weights
    include_bias=False,  # Add bias neuron to reservoir
    seed=42              # Random seed for reproducibility
)
```

## Configuring Readouts

The readout maps the high-dimensional reservoir state to the target output.

*   **Ridge Regression (`readouts.Ridge`)**: The standard offline training method. Fast and stable.
*   **Recursive Least Squares (`readouts.Rls`)**: For online, adaptive learning.
*   **Least Mean Squares (`readouts.Lms`)**: A simpler gradient-based online method.

## Building the Model

The `ESN` class acts as a container.

```python
model = ESN(connection_type="serial") # "serial" or "parallel"
model.add_reservoir(res1)
# For deep ESNs:
# model.add_reservoir(res2)
model.set_readout(readout)
```
