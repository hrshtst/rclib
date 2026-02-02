# Advanced Usage

## Online Learning

For real-time applications where data arrives sequentially, use `partial_fit` with an RLS or LMS readout.

```python
readout = readouts.Rls(lambda_=0.99, delta=1.0, include_bias=True)
model.set_readout(readout)

# In a loop:
model.partial_fit(x_step, y_step)
```

## Generative Prediction

To generate sequences autonomously (feeding predictions back as inputs):

```python
# Prime the reservoir with some initial data
prime_data = x_test[:100]
# Generate the next 200 steps
generated = model.predict_generative(prime_data, n_steps=200)
```

## Next-Generation RC (NVAR)

`rclib` supports NVAR, which uses time-delayed features instead of a random network.

```python
res = reservoirs.Nvar(num_lags=5)
# ... use as a normal reservoir
```

## Parallelization Configuration

You can optimize performance for your hardware by configuring CMake options during the build.

| Option | Default | Best For |
| :--- | :--- | :--- |
| `RCLIB_USE_OPENMP` | `ON` | Multi-core CPUs |
| `RCLIB_ENABLE_EIGEN_PARALLELIZATION` | `OFF` | Very large matrices (40k+ neurons) |

To change these, reinstall with:
```bash
CMAKE_ARGS="-DRCLIB_ENABLE_EIGEN_PARALLELIZATION=ON" pip install .
```
or via direct CMake configuration if building manually.
