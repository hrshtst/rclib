"""Readout configurations."""

from __future__ import annotations


class Ridge:
    """Ridge Regression Readout configuration."""

    def __init__(self, alpha: float, *, include_bias: bool) -> None:
        """Initialize the Ridge Readout.

        Args:
            alpha: Regularization parameter.
            include_bias: Whether to include a bias term.
        """
        self.alpha = alpha
        self.include_bias = include_bias


class Rls:
    """Recursive Least Squares (RLS) Readout configuration."""

    def __init__(self, lambda_: float, delta: float, *, include_bias: bool) -> None:
        """Initialize the RLS Readout.

        Args:
            lambda_: Forgetting factor (0.0 to 1.0).
            delta: Initial value for the covariance matrix diagonal.
            include_bias: Whether to include a bias term.
        """
        self.lambda_ = lambda_
        self.delta = delta
        self.include_bias = include_bias


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
