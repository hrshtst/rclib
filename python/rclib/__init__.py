"""Reservoir Computing Library (rclib)."""

from __future__ import annotations

from . import readouts, reservoirs
from .model import ESN

__all__ = ["ESN", "readouts", "reservoirs"]
