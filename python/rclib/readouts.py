"""Readout configurations."""

from __future__ import annotations


class Ridge:
    """Ridge Regression Readout configuration."""

    def __init__(
        self,
        alpha: float,
        *,
        include_bias: bool,
        solver: str = "auto",
        tolerance: float = 1e-10,
    ) -> None:
        """Initialize the Ridge Readout.

        Args:
            alpha: Regularization parameter.
            include_bias: Whether to include a bias term.
            solver: Solver to use ("auto", "cholesky", "dual_cholesky",
                "conjugate_gradient", "conjugate_gradient_implicit").
            tolerance: Convergence tolerance for iterative solvers (CG).
        """
        self.alpha = alpha
        self.include_bias = include_bias
        self.solver = solver
        self.tolerance = tolerance


class Rls:
    """Recursive Least Squares (RLS) Readout configuration."""

    def __init__(
        self,
        lambda_: float,
        delta: float,
        *,
        include_bias: bool,
        solver: str = "rank1_update",
    ) -> None:
        """Initialize the RLS Readout.

        Args:
            lambda_: Forgetting factor (0.0 to 1.0).
            delta: Initial value for the covariance matrix diagonal.
            include_bias: Whether to include a bias term.
            solver: Solver type ("rank1_update" or "rank_k_update").
                "rank1_update" is traditional sequential RLS.
                "rank_k_update" is optimized for mini-batches using Woodbury identity.
        """
        self.lambda_ = lambda_
        self.delta = delta
        self.include_bias = include_bias
        self.solver = solver


class Lms:
    """Least Mean Squares (LMS) Readout configuration."""

    def __init__(self, learning_rate: float, *, include_bias: bool) -> None:
        """Initialize the LMS Readout.

        Args:
            learning_rate: Learning rate for the LMS algorithm.
            include_bias: Whether to include a bias term.
        """
        self.learning_rate = learning_rate
        self.include_bias = include_bias
