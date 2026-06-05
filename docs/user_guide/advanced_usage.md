# Advanced Usage

## Online Learning

For real-time applications where data arrives sequentially, use `partial_fit` with an RLS or LMS readout. `rclib` supports **mini-batch** updates for both, providing significant speedups when processing multiple samples at once.

### Mini-batch LMS
The `Lms` readout automatically uses GEMM-based averaged batch-gradient updates
when `partial_fit` is called with multiple samples.

### Mini-batch RLS
The `Rls` readout provides two strategies for handling mini-batches:

| Solver | Strategy | Best For |
| :--- | :--- | :--- |
| `rank1_update` (Default) | Sequential Rank-1 updates | Single samples or small batches with $\lambda < 1.0$. |
| `rank_k_update` | Woodbury Rank-K update (GEMM) | Mini-batches (32+) with $\lambda = 1.0$. |

```python
# Create an RLS readout optimized for mini-batches
readout = readouts.Rls(
    lambda_=1.0,
    delta=1.0,
    include_bias=True,
    solver="rank_k_update"
)
model.set_readout(readout)

# In a loop (processing 64 samples at once):
for i in range(0, len(X), 64):
    model.partial_fit(X[i:i+64], Y[i:i+64])
```

> **Note:** `rank_k_update` is mathematically equivalent to sequential RLS only when the forgetting factor `lambda` is 1.0. For `lambda < 1.0`, `rclib` automatically falls back to sequential `rank1_update` to ensure mathematical correctness.

## Generative Prediction

To generate sequences autonomously (feeding predictions back as inputs):

```python
# Prime the reservoir with some initial data
prime_data = x_test[:100]
# Generate the next 200 steps
generated = model.predict_generative(prime_data, n_steps=200)
```

## Ridge Regression Solver Selection

`rclib` provides multiple strategies for batch training. While the `auto` mode is recommended, you can explicitly set the solver based on your specific needs.

```python
# Create a Ridge readout with an explicit solver
# Available: "auto", "cholesky", "dual_cholesky",
#            "conjugate_gradient", "conjugate_gradient_implicit"
readout = readouts.Ridge(
    alpha=1e-8,
    include_bias=True,
    solver="dual_cholesky"
)
```

| Solver | Best For |
| :--- | :--- |
| `cholesky` | Small reservoirs or when $N \le T$. |
| `dual_cholesky` | Large reservoirs with fewer samples ($N > T$). |
| `conjugate_gradient_implicit` | Extremely large reservoirs (`n_features >= 4,000`). |
| `auto` (Default) | Automatically chooses the most efficient strategy. |

## Next-Generation RC (NVAR)

`rclib` supports NVAR, which uses time-delayed polynomial features instead of a
random network. `polynomial_order=1` is a linear delay embedding; higher orders
append all monomials with replacement up to that degree.

```python
res = reservoirs.Nvar(num_lags=5, polynomial_order=2)
# ... use as a normal reservoir
```

## Parallelization Configuration

You can optimize performance for your hardware by configuring CMake options during the build.

| Option | Default | Best For |
| :--- | :--- | :--- |
| `RCLIB_USE_OPENMP` | `ON` | Multi-core CPUs |
| `RCLIB_ENABLE_EIGEN_PARALLELIZATION` | `ON` | Balanced performance (Default) |
| `RCLIB_ADAPTIVE_PARALLELIZATION` | `ON` | Automatic switching based on problem size (N > 1000) |

### Common Scenarios

**1. Default (Adaptive)**
Automatically uses serial mode for small reservoirs to avoid overhead and parallel mode for large ones.
```bash
CMAKE_ARGS="-DRCLIB_ADAPTIVE_PARALLELIZATION=ON" uv sync
```

**2. Forced Parallelism**
Force parallel execution even for small reservoirs.
```bash
CMAKE_ARGS="-DRCLIB_ADAPTIVE_PARALLELIZATION=OFF" uv sync
```

**3. Completely Serial**
Disable all multi-threading (best for debugging).
```bash
CMAKE_ARGS="-DRCLIB_USE_OPENMP=OFF" uv sync
```
