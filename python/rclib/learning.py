# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Configurations of reward-modulated Hebbian learning.

A reward-modulated component (``readouts.RewardModulated``) adds exploration noise to the
potentials of its post-synaptic units, correlates pre-synaptic activity with a post-synaptic
signal into an eligibility, and changes its weights in proportion to the eligibility and a
modulator derived from the reward. In the row-vector convention of rclib, with the weights of
shape ``(n_pre, n_post)``::

    S(y)  = sign(y) * |y| ** eligibility_power
    e     = trace_decay * e + S(pre).T @ S(post)        (update="online", every exploring row)
    E     = E + S(pre).T @ S(post)                       (update="trial", cleared by each update)
    dW    = eta * m * e,  eta * m * E,  or  eta * m * inv(C + decorrelation * I) @ E
    m     = R - R_avg  ("difference")  or  1 if R > R_avg else 0  ("heaviside")

where ``post`` is the injected noise, the potential minus its average before the row
(``"high_pass"``), or the potential itself, ``C`` sums ``pre.T @ pre`` over the trial, and
``R_avg`` is the reward baseline of the reward's key before that reward. See the user guide
for how the four published variants map to these settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Largest seed of the exploration noise, which the C++ core stores as an unsigned 32-bit int.
MAX_SEED = 2**32 - 1
#: Largest reward window, mirroring the C++ core.
MAX_BASELINE_WINDOW = 1_000_000


def _check_unit_interval(name: str, value: float, *, include_one: bool) -> None:
    # Chained range checks reject NaN on their own.
    if not (0 <= value <= 1 if include_one else 0 <= value < 1):
        msg = f"{name} must be in [0, {'1]' if include_one else '1)'}."
        raise ValueError(msg)


def _check_positive(name: str, value: float | None, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if value is None or not math.isfinite(value) or value <= 0:
        suffix = " when set" if optional else ""
        msg = f"{name} must be finite and positive{suffix}."
        raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class Exploration:
    """Exploration noise added to the potentials of the post-synaptic units.

    Attributes
    ----------
    distribution : str
        ``"uniform"`` on ``[-amplitude, amplitude]`` or ``"gaussian"`` with standard deviation
        ``amplitude``.
    amplitude : float
        Scale of the noise; 0 turns exploration off, which stops learning.
    probability : float
        Probability in ``(0, 1]`` that a unit is perturbed in a row; 1 perturbs every unit in
        every row, and small values give sparse kicks.
    seed : int
        Seed of the noise in ``[0, 2**32 - 1]``. Components that share a rule draw separate noise.
    """

    distribution: str = "uniform"
    amplitude: float = 0.1
    probability: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises
        ------
        ValueError
            If a setting is out of range.
        """
        if self.distribution not in {"uniform", "gaussian"}:
            msg = "distribution must be 'uniform' or 'gaussian'."
            raise ValueError(msg)
        if not math.isfinite(self.amplitude) or self.amplitude < 0:
            msg = "amplitude must be finite and non-negative."
            raise ValueError(msg)
        if not 0 < self.probability <= 1:
            msg = "probability must be in (0, 1]."
            raise ValueError(msg)
        if not 0 <= self.seed <= MAX_SEED:
            msg = f"seed must be in [0, {MAX_SEED}]."
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class Baseline:
    """The reward baseline ``R_avg`` a reward is compared with, kept per integer key.

    The first reward of a key with an ``"ema"`` or ``"window"`` baseline becomes its own baseline,
    so its modulator is zero.

    Attributes
    ----------
    kind : str
        ``"ema"``: ``R_avg = decay * R_avg + (1 - decay) * R`` after each reward; ``"window"``: the
        mean of the last ``window`` rewards; ``"none"``: 0, so the modulator uses the reward itself.
    decay : float
        Decay in ``[0, 1)`` of the ``"ema"`` baseline.
    window : int
        Number of rewards in ``[1, 1000000]`` the ``"window"`` baseline averages.
    """

    kind: str = "ema"
    decay: float = 0.8
    window: int = 50

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises
        ------
        ValueError
            If a setting is out of range.
        """
        if self.kind not in {"ema", "window", "none"}:
            msg = "kind must be 'ema', 'window' or 'none'."
            raise ValueError(msg)
        _check_unit_interval("decay", self.decay, include_one=False)
        if not 1 <= self.window <= MAX_BASELINE_WINDOW:
            msg = f"window must be in [1, {MAX_BASELINE_WINDOW}]."
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class RewardModulatedHebbian:
    """A reward-modulated Hebbian rule.

    Attributes
    ----------
    learning_rate : float
        Learning rate ``eta`` at the first update.
    exploration : Exploration
        The exploration noise.
    post : str
        Post-synaptic signal: ``"noise"`` (the injected noise, node perturbation),
        ``"high_pass"`` (the potential minus its average before the row) or ``"potential"``.
    post_decay : float
        Decay in ``[0, 1)`` of the potential's average for ``"high_pass"``; 0 makes the signal
        the first difference of the potential.
    eligibility_power : float
        Exponent ``p > 0`` of ``S(y) = sign(y) |y|^p``: 1 is linear, 3 the supralinear cube.
    update : str
        ``"online"``: the eligibility decays by ``trace_decay`` every exploring row and each
        update uses it; ``"trial"``: the eligibility sums the rows of a trial, and each update uses
        and clears it.
    trace_decay : float
        Decay in ``[0, 1]`` of the online eligibility; must be 0 with ``update="trial"``.
    modulator : str
        ``"difference"`` (``R - R_avg``) or ``"heaviside"`` (1 if ``R > R_avg``, else 0).
    baseline : Baseline
        The reward baseline.
    learning_rate_decay_steps : float, optional
        ``T`` in ``eta_n = learning_rate / (1 + n / T)``, ``n`` the number of updates so far.
    max_update : float, optional
        Clip of each weight change to ``[-max_update, max_update]``.
    decorrelation : float, optional
        Ridge term ``lambda > 0`` of the decorrelated update ``inv(C + lambda I) @ E``; requires
        ``update="trial"`` and an ``eligibility_power`` of 1.
    """

    learning_rate: float
    exploration: Exploration = field(default_factory=Exploration)
    post: str = "high_pass"
    post_decay: float = 0.0
    eligibility_power: float = 1.0
    update: str = "online"
    trace_decay: float = 0.0
    modulator: str = "difference"
    baseline: Baseline = field(default_factory=Baseline)
    learning_rate_decay_steps: float | None = None
    max_update: float | None = None
    decorrelation: float | None = None

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises
        ------
        ValueError
            If a setting is out of range or settings conflict.
        TypeError
            If ``exploration`` or ``baseline`` has the wrong type.
        """
        _check_positive("learning_rate", self.learning_rate)
        if not isinstance(self.exploration, Exploration):
            msg = "exploration must be an Exploration."
            raise TypeError(msg)
        if not isinstance(self.baseline, Baseline):
            msg = "baseline must be a Baseline."
            raise TypeError(msg)
        if self.post not in {"noise", "high_pass", "potential"}:
            msg = "post must be 'noise', 'high_pass' or 'potential'."
            raise ValueError(msg)
        _check_unit_interval("post_decay", self.post_decay, include_one=False)
        _check_positive("eligibility_power", self.eligibility_power)
        if self.update not in {"online", "trial"}:
            msg = "update must be 'online' or 'trial'."
            raise ValueError(msg)
        _check_unit_interval("trace_decay", self.trace_decay, include_one=True)
        if self.update == "trial" and self.trace_decay != 0:
            msg = "trace_decay applies to online updates only; trial eligibility never decays."
            raise ValueError(msg)
        if self.modulator not in {"difference", "heaviside"}:
            msg = "modulator must be 'difference' or 'heaviside'."
            raise ValueError(msg)
        _check_positive("learning_rate_decay_steps", self.learning_rate_decay_steps, optional=True)
        _check_positive("max_update", self.max_update, optional=True)
        _check_positive("decorrelation", self.decorrelation, optional=True)
        if self.decorrelation is not None and self.update != "trial":
            msg = "decorrelation requires trial updates."
            raise ValueError(msg)
        if self.decorrelation is not None and self.eligibility_power != 1:
            msg = "decorrelation requires an eligibility_power of 1."
            raise ValueError(msg)
