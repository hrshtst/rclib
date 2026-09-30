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
readout = readouts.Rls(lambda_=1.0, delta=1.0, include_bias=True, solver="rank_k_update")
model.set_readout(readout)

# In a loop (processing 64 samples at once):
for i in range(0, len(X), 64):
    model.partial_fit(X[i : i + 64], Y[i : i + 64])
```

> **Note:** `rank_k_update` is mathematically equivalent to sequential RLS only when the forgetting factor `lambda` is 1.0. For `lambda < 1.0`, `rclib` automatically falls back to sequential `rank1_update` to ensure mathematical correctness.

## Generative Prediction

To generate sequences autonomously (feeding predictions back as inputs):

```python
# Prime the reservoir with some initial data
prime_data = x_test[:100]
# Generate the next 200 steps
generated = model.predict_generative(prime_data, n_steps=200)
# Continue the same sequence for another 100 steps
more = model.predict_generative(np.empty((0, 1)), n_steps=100)
```

Every generated output, including the last, is fed back into the reservoirs, so a
call with empty priming data continues where the previous call stopped:
generating 200 and then 100 steps gives the same outputs and final reservoir
states as generating 300 steps at once. The readout's output width must match the
model's input width, since each output becomes the next input.

## Spectral Radius Scaling

`RandomSparse` draws a sparse random matrix $\mathbf{W}_{res}$ from its `seed` and
scales it so that its spectral radius, the largest eigenvalue modulus, equals
`spectral_radius`. `spectral_radius_method` chooses how the spectral radius of
the unscaled matrix is found:

| Method | Observed error of the scaled radius | Cost | Best For |
| :--- | :--- | :--- | :--- |
| `power_iteration` (Default) | Mean about 0.01%, worst 0.75% (200 seeds per size, 20 to 1000 neurons; not a bound) | 1000 sparse matrix-vector products | Any size |
| `dense` | Exact up to rounding | $O(n^3)$ time, $O(n^2)$ memory | Small reservoirs, or when the radius must be exact |

```python
res = reservoirs.RandomSparse(n_neurons=200, spectral_radius=0.95, seed=0, spectral_radius_method="dense")
```

### How the default works

Plain power iteration multiplies a start vector by $\mathbf{W}_{res}$ over and
over, normalizing it each time, and reads the spectral radius off the norm of
the last step. That converges when one real eigenvalue is clearly larger in
modulus than all others, as for many symmetric matrices. The eigenvalues of a
non-symmetric random matrix instead fill a disk, so the largest ones are often a
complex-conjugate pair or several eigenvalues of nearly equal modulus. The
iterate then keeps rotating among them and the norm of the last step oscillates
instead of converging. Measured over 50 seeds at sparsity 0.1, plain power
iteration with 100 steps missed the requested radius by up to 11% for 100
neurons, 6% for 300 and 4% for 1000, and more steps did not fix it.

`rclib` therefore averages. It runs 1000 steps from a start vector drawn from a
generator seeded by `seed`, and returns the geometric mean of the per-step growth
$\lVert \mathbf{W}_{res} \mathbf{b}_k \rVert / \lVert \mathbf{b}_k \rVert$ over the
last 500. The first 500 steps let the dominant eigenvalues take over, and
averaging over many steps cancels their oscillation. Since the start vector
depends only on `seed`, reservoirs built with the same parameters get the same
weights, whatever else the program does.

The averaged estimate is still an estimate. Measured over 200 seeds for each of
20, 30, 50, 100, 300 and 1000 neurons at sparsity 0.1 and a requested radius of
0.9, the scaled matrix missed the requested radius by 0.01% to 0.02% on average.
The worst seed missed by 0.75% (100 neurons); at 30 neurons, seed 113 missed by
0.47%. These figures describe that sample and are not bounds: another size,
sparsity or seed can miss by more. Use `spectral_radius_method="dense"` when the
radius must be exact.

Each step is one sparse matrix-vector product, so the cost grows with the number
of non-zero weights, `sparsity * n_neurons**2`. Single-threaded on the
development machine at sparsity 0.1, the estimate took about 0.4 ms for 100
neurons, 3 ms for 300, 27 ms for 1000 and 2 s for 10,000. The `dense` method took
about 1.4 ms for 100 neurons, 30 ms for 300 and 1.2 s for 1000, growing as
$n^3$.

> **Note:** rclib 0.2.0 and earlier used plain power iteration from a start
> vector drawn from the global `std::rand()` state, so only the first reservoir
> built in a process was reproducible. Since the averaged estimate, the same
> `seed` gives a slightly differently scaled $\mathbf{W}_{res}$ than in those
> versions. Saved models keep the weights they were saved with.

## Ridge Regression Solver Selection

`rclib` provides multiple strategies for batch training. While the `auto` mode is recommended, you can explicitly set the solver based on your specific needs.

```python
# Create a Ridge readout with an explicit solver
# Available: "auto", "cholesky", "dual_cholesky",
#            "conjugate_gradient", "conjugate_gradient_implicit"
readout = readouts.Ridge(alpha=1e-8, include_bias=True, solver="dual_cholesky")
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

## Saving and Loading Models

A model can be written to a file and restored later, from Python or from C++.
Files are interchangeable between the two, because both go through the same C++
code.

```python
from rclib import ESN

model.fit(x_train, y_train, washout_len=100)
model.save("mackey_glass.rclib")

restored = ESN.load("mackey_glass.rclib")
y_pred = restored.predict(x_test)
```

```cpp
model.save("mackey_glass.rclib");
Model restored = Model::load("mackey_glass.rclib");
```

The file holds the configuration, the trained weights (the random reservoir
matrices themselves, not just their seed), the RLS/LMS training state and the
current reservoir states. A restored model therefore continues where the
original stopped: `predict_online`, `partial_fit`, and `predict_generative`
without priming data all behave as they would have on the original. `ESN`
objects also support `pickle` and `copy.deepcopy`, which use the same format.

Saving replaces an existing file in one step: if saving fails, the existing
file is left unchanged. Every save or load failure raises
`rclib.SerializationError` (a `RuntimeError` subclass; `SerializationError` in
C++). Examples are a model without a readout, a component whose widths do not
match its neighbours, a custom reservoir or readout type, the same reservoir
object added twice (C++ only), and a corrupted or truncated file. Running out of
memory raises `MemoryError` (`std::bad_alloc` in C++) instead.

> **Compatibility:** Files written by rclib 0.2.0 still load; their `RandomSparse`
> reservoirs report `spectral_radius_method="power_iteration"`. Files written by
> later versions use model format version 2 and cannot be read by rclib 0.2.0.

> **Security:** Only load files from sources you trust. Unlike `pickle`, the
> format contains no executable code, but it is parsed by native code. Loading a
> pickled `ESN` is as unsafe as any other unpickling.

> **Portability:** On the same build and settings, a restored model produces
> bit-identical results. Files load on every supported (little-endian) platform
> and the stored values are exact, but computations on a different compiler, CPU
> or thread configuration can round differently, and nothing bounds how far such
> differences grow: recurrent updates, generative rollouts and continued training
> can amplify them. A `RandomSparse` reservoir that has never received input has
> no input weights yet; they are generated from the seed on first use and can
> differ across platforms, so fit the model before saving if you need portable
> weights.

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
