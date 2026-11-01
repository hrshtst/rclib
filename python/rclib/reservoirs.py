# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Reservoir configurations."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable

from .learning import RewardModulatedHebbian

# Upper bound on the NVAR polynomial order, mirroring NvarReservoir::max_polynomial_order
# in the C++ core. Bounds monomial-generation recursion depth.
MAX_POLYNOMIAL_ORDER = 32


def _check_random_sparse(
    n_neurons: int,
    spectral_radius: float,
    sparsity: float,
    leak_rate: float,
    input_scaling: float,
    spectral_radius_method: str,
    bias_scaling: float,
) -> None:
    """Check the hyperparameters of a reservoir with random sparse recurrence, as the C++ core does."""
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
    if not math.isfinite(bias_scaling) or bias_scaling < 0:
        msg = "bias_scaling must be finite and non-negative."
        raise ValueError(msg)


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
        bias_scaling: float = 1.0,
    ) -> None:
        """Initialize the Random Sparse Reservoir.

        Args:
            n_neurons: Number of neurons in the reservoir.
            spectral_radius: Spectral radius of the reservoir weight matrix.
            sparsity: Sparsity of the reservoir weight matrix (0.0 to 1.0).
            leak_rate: Leaking rate of the neurons.
            input_scaling: Scaling factor for the input weights.
            include_bias: Whether to include a bias term: one constant input per neuron, drawn
                uniformly from [-1, 1] and multiplied by ``bias_scaling``.
            seed: Random seed for weights initialization.
            spectral_radius_method: How the spectral radius of the random weight matrix is
                found before it is scaled to ``spectral_radius``. "power_iteration" (default)
                estimates it by seeded power iteration, typically to within 0.3%. "dense"
                computes it exactly from all eigenvalues of the matrix, which costs
                O(n_neurons^3) time and O(n_neurons^2) memory, so it is meant for small
                reservoirs.
            bias_scaling: Scaling factor for the bias, set apart from ``input_scaling``: the bias
                lies in [-bias_scaling, bias_scaling]. The draws do not depend on it, so the same
                seed gives the same recurrent and input weights at every bias_scaling, and 1
                (default) gives the unscaled bias. Without ``include_bias`` it has no effect.
        """
        _check_random_sparse(
            n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, spectral_radius_method, bias_scaling
        )

        self.n_neurons = n_neurons
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.include_bias = include_bias
        self.seed = seed
        self.spectral_radius_method = spectral_radius_method
        self.bias_scaling = bias_scaling


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


def _normalize_mask(name: str, neurons: Iterable[int] | None, n_neurons: int) -> tuple[int, ...] | None:
    """Sort a list of neuron indices, checking that they are distinct and in range; None means all."""
    if neurons is None:
        return None
    indices = tuple(sorted(operator.index(neuron) for neuron in neurons))
    if len(set(indices)) != len(indices):
        msg = f"{name} must list distinct neurons."
        raise ValueError(msg)
    if indices and (indices[0] < 0 or indices[-1] >= n_neurons):
        msg = f"{name} must hold neuron indices in [0, n_neurons)."
        raise ValueError(msg)
    return indices


class RewardModulated:
    """Reward-modulated reservoir configuration: random sparse recurrence that learns from rewards.

    The neurons integrate a potential ``v`` and fire at the rate ``r = tanh(v)``::

        v = (1 - leak_rate) * v + leak_rate * (u @ W_in + bias + r_prev @ W_res) + noise

    where the exploration noise of the rule enters only exploring steps (``ESN.act``), outside the
    leak. ``ESN.reinforce`` then changes the existing recurrent synapses from the neurons in
    ``plastic_presynaptic`` to those in ``plastic_postsynaptic``. With ``leak_rate`` 1 and no noise
    this is a ``RandomSparse`` reservoir with the same weights. A model without a readout returns
    the rates from ``ESN.act``, so designated neurons can serve as outputs.
    """

    def __init__(
        self,
        n_neurons: int,
        spectral_radius: float,
        sparsity: float = 0.1,
        leak_rate: float = 1.0,
        input_scaling: float = 1.0,
        *,
        rule: RewardModulatedHebbian,
        include_bias: bool = False,
        seed: int = 42,
        spectral_radius_method: str = "power_iteration",
        bias_scaling: float = 1.0,
        plastic_presynaptic: Iterable[int] | None = None,
        plastic_postsynaptic: Iterable[int] | None = None,
    ) -> None:
        """Initialize the reward-modulated reservoir.

        Args:
            n_neurons: Number of neurons in the reservoir.
            spectral_radius: Spectral radius of the initial recurrent weights.
            sparsity: Sparsity of the recurrent weights (0.0 to 1.0); the pattern stays fixed.
            leak_rate: Leak rate of the potential, in (0, 1].
            input_scaling: Scaling factor for the input weights.
            rule: The learning rule, a ``learning.RewardModulatedHebbian``.
            include_bias: Whether to include a bias, as in ``RandomSparse``.
            seed: Random seed for the weights, as in ``RandomSparse``.
            spectral_radius_method: "power_iteration" or "dense", as in ``RandomSparse``.
            bias_scaling: Scaling factor for the bias, as in ``RandomSparse``.
            plastic_presynaptic: Neurons whose outgoing synapses learn; None means all.
            plastic_postsynaptic: Neurons whose incoming synapses learn; None means all.
        """
        _check_random_sparse(
            n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, spectral_radius_method, bias_scaling
        )
        if not isinstance(rule, RewardModulatedHebbian):
            msg = "rule must be a learning.RewardModulatedHebbian."
            raise TypeError(msg)

        self.n_neurons = n_neurons
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.rule = rule
        self.include_bias = include_bias
        self.seed = seed
        self.spectral_radius_method = spectral_radius_method
        self.bias_scaling = bias_scaling
        self.plastic_presynaptic = _normalize_mask("plastic_presynaptic", plastic_presynaptic, n_neurons)
        self.plastic_postsynaptic = _normalize_mask("plastic_postsynaptic", plastic_postsynaptic, n_neurons)
