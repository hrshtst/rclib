# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Reservoir Computing Library (rclib)."""

from __future__ import annotations

from . import _rclib, readouts, reservoirs
from .model import ESN

#: Raised when a model cannot be saved or a model file cannot be loaded (a RuntimeError subclass).
SerializationError: type[RuntimeError] = _rclib.SerializationError

__all__ = ["ESN", "SerializationError", "readouts", "reservoirs"]
