# Readout Algorithms

The readout layer maps the reservoir state $\mathbf{x}(t)$ to the output $\mathbf{y}(t)$.

$$ \mathbf{y}(t) = \mathbf{W}_{out} \mathbf{x}(t) $$

## Ridge Regression (Batch)

Ridge regression (also known as Tikhonov regularization) minimizes the squared error while penalizing large weights to prevent overfitting. `rclib` implements several strategies to solve this efficiently.

### Primal Formulation ($N \le T$)

When the number of neurons ($N$) is less than or equal to the number of samples ($T$), we solve the normal equations:

$$ \mathbf{W}_{out} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y} $$

Where $\mathbf{X}$ is the $T \times N$ state matrix, $\mathbf{Y}$ is the $T \times O$ target matrix, and $\alpha$ is the regularization parameter. `rclib` uses optimized matrix-matrix multiplication (GEMM) to form the $N \times N$ covariance matrix $\mathbf{X}^T \mathbf{X}$ efficiently, leveraging multi-core parallelization.

### Dual Formulation ($N > T$)

When the reservoir is very large ($N > T$), the dual formulation is more efficient as it operates in the $T \times T$ sample space:

$$ \mathbf{W}_{out} = \mathbf{X}^T (\mathbf{X} \mathbf{X}^T + \alpha \mathbf{I})^{-1} \mathbf{Y} $$

This approach significantly reduces computational complexity from $O(N^3)$ to $O(T^3)$ in underdetermined cases.

### Adaptive Solver Selection

`rclib` automatically selects the optimal solver based on the problem dimensions:
- **Primal Cholesky**: Standard case ($N \le T$).
- **Dual Cholesky**: High-dimensional case ($N > T$).
- **Implicit Conjugate Gradient**: Very high-dimensional case ($N \ge 8,000$) where explicit matrix formation is avoided.

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

## Least Mean Squares (LMS) (Online)

LMS is a stochastic gradient descent method.

$$ \mathbf{W} \leftarrow \mathbf{W} + \eta \mathbf{e} \mathbf{x}^T $$

Where $\eta$ is the learning rate. It is computationally cheaper ($O(N)$) than RLS ($O(N^2)$) but typically converges slower.
