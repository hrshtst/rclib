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
        self.n_neurons = n_neurons
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.include_bias = include_bias
        self.seed = seed


class Nvar:
    """NVAR Reservoir configuration."""

    def __init__(self, num_lags: int) -> None:
        """Initialize the NVAR Reservoir.

        Args:
            num_lags: Number of time lags to include.
        """
        self.num_lags = num_lags
