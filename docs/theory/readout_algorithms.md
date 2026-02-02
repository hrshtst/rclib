# Readout Algorithms

The readout layer maps the reservoir state $\mathbf{x}(t)$ to the output $\mathbf{y}(t)$.

$$ \mathbf{y}(t) = \mathbf{W}_{out} \mathbf{x}(t) $$

## Ridge Regression (Batch)

For batch training, we solve for $\mathbf{W}_{out}$ that minimizes the squared error with $L_2$ regularization:

$$ \mathbf{W}_{out} = \mathbf{Y}_{target} \mathbf{X}^T (\mathbf{X}\mathbf{X}^T + \beta \mathbf{I})^{-1} $$

Where $\mathbf{X}$ collects all state vectors over time, and $\beta$ is the regularization parameter.

## Recursive Least Squares (RLS) (Online)

RLS updates the weights recursively for each new data point. It maintains an inverse covariance matrix $\mathbf{P}$.

1.  **Gain Calculation:**
    $$ \mathbf{k} = \frac{\mathbf{P} \mathbf{x}}{ \lambda + \mathbf{x}^T \mathbf{P} \mathbf{x} } $$
2.  **Weight Update:**
    $$ \mathbf{W} \leftarrow \mathbf{W} + \mathbf{k} \mathbf{e}^T $$
    where $\mathbf{e} = \mathbf{d} - \mathbf{W}^T \mathbf{x}$ is the prediction error.
3.  **Covariance Matrix Update:**
    $$ \mathbf{P} \leftarrow \lambda^{-1} (\mathbf{P} - \mathbf{k} \mathbf{x}^T \mathbf{P}) $$

`rclib` implements an optimized version of the covariance update using symmetric rank-1 updates to significantly reduce computational cost:

$$ \mathbf{P} \leftarrow \lambda^{-1} \left( \mathbf{P} - \frac{(\mathbf{P}\mathbf{x})(\mathbf{P}\mathbf{x})^T}{\lambda + \mathbf{x}^T \mathbf{P} \mathbf{x}} \right) $$

## Least Mean Squares (LMS) (Online)

LMS is a stochastic gradient descent method.

$$ \mathbf{W} \leftarrow \mathbf{W} + \eta \mathbf{e} \mathbf{x}^T $$

Where $\eta$ is the learning rate. It is computationally cheaper ($O(N)$) than RLS ($O(N^2)$) but typically converges slower.
