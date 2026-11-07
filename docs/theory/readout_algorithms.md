# Readout Algorithms

The readout layer maps the reservoir state $\mathbf{x}(t)$ to the output $\mathbf{y}(t)$.

$$ \mathbf{y}(t) = \mathbf{W}_{out} \mathbf{x}(t) $$

## Ridge Regression (Batch)

Ridge regression (also known as Tikhonov regularization) minimizes the squared error while penalizing large weights to prevent overfitting. `rclib` implements several strategies to solve this efficiently.

### Primal Formulation ($N \le T$) { data-toc-label="Primal Formulation (N ≤ T)" }

When the number of neurons ($N$) is less than or equal to the number of samples ($T$), we solve the normal equations:

$$ \mathbf{W}_{out} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y} $$

Where $\mathbf{X}$ is the $T \times N$ state matrix, $\mathbf{Y}$ is the $T \times O$ target matrix, and $\alpha$ is the regularization parameter. `rclib` uses optimized matrix-matrix multiplication (GEMM) to form the $N \times N$ covariance matrix $\mathbf{X}^T \mathbf{X}$ efficiently, leveraging multi-core parallelization.

### Dual Formulation ($N > T$) { data-toc-label="Dual Formulation (N > T)" }

When the reservoir is very large ($N > T$), the dual formulation is more efficient as it operates in the $T \times T$ sample space:

$$ \mathbf{W}_{out} = \mathbf{X}^T (\mathbf{X} \mathbf{X}^T + \alpha \mathbf{I})^{-1} \mathbf{Y} $$

This approach significantly reduces computational complexity from $O(N^3)$ to $O(T^3)$ in underdetermined cases.

### Adaptive Solver Selection

`rclib` automatically selects a solver based on the problem dimensions:

- **Primal Cholesky**: Standard case ($N \le T$).
- **Dual Cholesky**: High-dimensional case ($N > T$).
- **Implicit Conjugate Gradient**: Very high-dimensional case
  (`n_features >= 4,000`) where explicit matrix formation is avoided.

When `include_bias=True`, `rclib` appends a constant feature to the state matrix.
The current ridge implementation regularizes this appended bias weight in the
same way as the other weights.

## Recursive Least Squares (RLS) (Online)

RLS updates the weights recursively for each new data point. It maintains an inverse covariance matrix $\mathbf{P}$.

1.  **Gain Calculation:**

    $$ \mathbf{k} = \frac{\mathbf{P} \mathbf{x}}{ \lambda + \mathbf{x}^T \mathbf{P} \mathbf{x} } $$

2.  **Weight Update:**

    $$ \mathbf{W} \leftarrow \mathbf{W} + \mathbf{k} \mathbf{e}^T $$

    where $\mathbf{e} = \mathbf{d} - \mathbf{W}^T \mathbf{x}$ is the prediction error.

3.  **Covariance Matrix Update:**

    $$ \mathbf{P} \leftarrow \lambda^{-1} (\mathbf{P} - \mathbf{k} \mathbf{x}^T \mathbf{P}) $$

`rclib` implements two strategies for the covariance update:

1. **Sequential Rank-1 Update:** For single samples or small batches, it uses symmetric rank-1 updates to reduce computational cost:

    $$ \mathbf{P} \leftarrow \lambda^{-1} \left( \mathbf{P} - \frac{(\mathbf{P}\mathbf{x})(\mathbf{P}\mathbf{x})^T}{\lambda + \mathbf{x}^T \mathbf{P} \mathbf{x}} \right) $$

2. **Woodbury Rank-K Update (Mini-batch):** For larger mini-batches ($\lambda = 1.0$), it leverages the Woodbury Matrix Identity to update the covariance matrix using matrix-matrix products (GEMM):

    $$ \mathbf{P}_{new} = \mathbf{P} - \mathbf{P} \mathbf{X}^T (\mathbf{I} + \mathbf{X} \mathbf{P} \mathbf{X}^T)^{-1} \mathbf{X} \mathbf{P} $$

    This turns sequential $O(B \cdot N^2)$ operations into high-density GEMM calls, significantly improving throughput on multi-core systems when the batch size $B$ is sufficiently large.

For `lambda < 1.0`, `rank_k_update` falls back to sequential rank-1 updates so
the result remains mathematically equivalent to ordinary RLS.

### Forgetting Methods

With $\lambda < 1$, each sample forgets part of the old information, so that the
weights can follow a target that changes. In terms of the information matrix
$\mathbf{R} = \mathbf{P}^{-1}$, the update above forgets *exponentially*: it
discounts $\mathbf{R}$ in every direction, then adds the sample,

$$ \mathbf{R} \leftarrow \lambda \mathbf{R} + \mathbf{x}\mathbf{x}^T. $$

In a direction that the inputs never excite, nothing is added, so $\mathbf{R}$
decays there as $\lambda^n$ and $\mathbf{P}$ grows as $\lambda^{-n}$: the
covariance *winds up*. This happens whenever the inputs span fewer dimensions
than the readout has features, for instance when every sample comes from the same
repeated trajectory, or when some features are linear combinations of others.
Rounding errors along those directions grow with $\mathbf{P}$, and the weights
are lost well before $\mathbf{P}$ overflows.

`forgetting_method="directional"` forgets only along the sample's input, following
the idea of directional forgetting (Kulhavý, 1987; Bittanti, Bolzern and Campi,
1990). Up to the noise variance, $\mathbf{x}^T \mathbf{P} \mathbf{x}$ is the
variance of the prediction for the input $\mathbf{x}$, so
$\mathbf{x}\mathbf{x}^T / (\mathbf{x}^T \mathbf{P} \mathbf{x})$ is the information
that $\mathbf{R}$ holds along $\mathbf{x}$: taking all of it away leaves a singular
matrix, with $\mathbf{P}\mathbf{x}$ in its null space. Directional forgetting takes
away the share $1 - \lambda$ of it, then adds the sample, as exponential forgetting
does:

$$ \mathbf{R} \leftarrow \mathbf{R} - (1 - \lambda) \frac{\mathbf{x}\mathbf{x}^T}{\mathbf{x}^T \mathbf{P} \mathbf{x}} + \mathbf{x}\mathbf{x}^T. $$

The update changes $\mathbf{R}$ only along the sample, so $\mathbf{R}\mathbf{v}$
stays as it was for every $\mathbf{v}$ orthogonal to all the inputs: directions
that the samples do not excite keep their information, and $\mathbf{P}$ stays
bounded there, while the excited ones still forget. By the Sherman–Morrison
formula, with $s = \mathbf{x}^T \mathbf{P} \mathbf{x}$,

$$ \mathbf{P} \leftarrow \mathbf{P} + \frac{(1 - \lambda) - s}{\lambda + s} \frac{(\mathbf{P}\mathbf{x})(\mathbf{P}\mathbf{x})^T}{s}, $$

one symmetric rank-1 update, as for exponential forgetting, so the cost is the
same. The updated $\mathbf{P}$ maps $\mathbf{x}$ to the same gain
$\mathbf{k} = \mathbf{P}\mathbf{x} / (\lambda + s)$, so the weights move by the same
formula. A few cases follow:

- at $\lambda = 1$ nothing is forgotten, and both methods are ordinary RLS;
- with a single feature, forgetting along the input is forgetting everywhere, and
  both methods agree;
- a zero input ($s = 0$) carries no information, and changes nothing.

Directional forgetting does not discount information about directions the inputs
no longer excite. If the target changes along such a direction, the readout
follows only once the inputs excite it again.

## Least Mean Squares (LMS) (Online)

LMS is a stochastic gradient descent method.

$$ \mathbf{W} \leftarrow \mathbf{W} + \eta \mathbf{e} \mathbf{x}^T $$

Where $\eta$ is the learning rate. It is computationally cheaper ($O(N)$) than RLS ($O(N^2)$) but typically converges slower.

When `partial_fit` receives more than one sample, `rclib` performs an averaged
batch-gradient LMS update:

$$ \mathbf{W} \leftarrow \mathbf{W} + \frac{\eta}{B}\mathbf{X}^T(\mathbf{Y} - \mathbf{X}\mathbf{W}) $$

where $B$ is the mini-batch size.

## References

- R. Kulhavý. Restricted exponential forgetting in real-time identification.
  *Automatica*, 23(5):589–600, 1987.
- S. Bittanti, P. Bolzern and M. Campi. Convergence and exponential convergence of
  identification algorithms with directional forgetting factor. *Automatica*,
  26(5):929–932, 1990.
