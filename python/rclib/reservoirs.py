# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Reservoir configurations."""

from __future__ import annotations

import math

# Upper bound on the NVAR polynomial order, mirroring NvarReservoir::max_polynomial_order
# in the C++ core. Bounds monomial-generation recursion depth.
MAX_POLYNOMIAL_ORDER = 32


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
        spectral_radius_method: str = "power_iteration",
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
            spectral_radius_method: How the spectral radius of the random weight matrix is
                found before it is scaled to ``spectral_radius``. "power_iteration" (default)
                estimates it by seeded power iteration, typically to within 0.3%. "dense"
                computes it exactly from all eigenvalues of the matrix, which costs
                O(n_neurons^3) time and O(n_neurons^2) memory, so it is meant for small
                reservoirs.
        """
        if n_neurons <= 0:
            msg = "n_neurons must be positive."
            raise ValueError(msg)
        # Chained range checks reject NaN on their own; one-sided bounds also need isfinite.
        if not math.isfinite(spectral_radius) or spectral_radius < 0:
            msg = "spectral_radius must be finite and non-negative."
            raise ValueError(msg)
        if not 0 <= sparsity <= 1:
            msg = "sparsity must be in [0, 1]."
            raise ValueError(msg)
        if not 0 < leak_rate <= 1:
            msg = "leak_rate must be in (0, 1]."
            raise ValueError(msg)
        if not math.isfinite(input_scaling) or input_scaling < 0:
            msg = "input_scaling must be finite and non-negative."
            raise ValueError(msg)
        if spectral_radius_method not in {"power_iteration", "dense"}:
            msg = "spectral_radius_method must be 'power_iteration' or 'dense'."
            raise ValueError(msg)

        self.n_neurons = n_neurons
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.include_bias = include_bias
        self.seed = seed
        self.spectral_radius_method = spectral_radius_method


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
        if polynomial_order > MAX_POLYNOMIAL_ORDER:
            msg = f"polynomial_order exceeds the supported maximum ({MAX_POLYNOMIAL_ORDER})."
            raise ValueError(msg)

        self.num_lags = num_lags
        self.polynomial_order = polynomial_order
