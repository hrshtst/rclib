"""Reservoir configurations."""

from __future__ import annotations


class RandomSparse:
    """Random Sparse Reservoir configuration."""

    def __init__(
        self,
        n_neurons: int,
        spectral_radius: float,
        sparsity: float = 0.1,
        leak_rate: float = 1.0,
        input_scaling: float = 1.0,
        *,
        include_bias: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize the Random Sparse Reservoir.

        Args:
            n_neurons: Number of neurons in the reservoir.
            spectral_radius: Spectral radius of the reservoir weight matrix.
            sparsity: Sparsity of the reservoir weight matrix (0.0 to 1.0).
            leak_rate: Leaking rate of the neurons.
            input_scaling: Scaling factor for the input weights.
            include_bias: Whether to include a bias term.
            seed: Random seed for weights initialization.
        """
        if n_neurons <= 0:
            msg = "n_neurons must be positive."
            raise ValueError(msg)
        if spectral_radius < 0:
            msg = "spectral_radius must be non-negative."
            raise ValueError(msg)
        if not 0 <= sparsity <= 1:
            msg = "sparsity must be in [0, 1]."
            raise ValueError(msg)
        if not 0 < leak_rate <= 1:
            msg = "leak_rate must be in (0, 1]."
            raise ValueError(msg)
        if input_scaling < 0:
            msg = "input_scaling must be non-negative."
            raise ValueError(msg)

        self.n_neurons = n_neurons
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.include_bias = include_bias
        self.seed = seed


class Nvar:
    """NVAR Reservoir configuration."""

    def __init__(self, num_lags: int, polynomial_order: int = 1) -> None:
        """Initialize the NVAR Reservoir.

        Args:
            num_lags: Number of time lags to include.
            polynomial_order: Maximum monomial degree to include. The default
                of 1 preserves a linear delay embedding.
        """
        if num_lags <= 0:
            msg = "num_lags must be positive."
            raise ValueError(msg)
        if polynomial_order <= 0:
            msg = "polynomial_order must be positive."
            raise ValueError(msg)

        self.num_lags = num_lags
        self.polynomial_order = polynomial_order
