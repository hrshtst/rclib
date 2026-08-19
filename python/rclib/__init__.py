# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Reservoir Computing Library (rclib)."""

from __future__ import annotations

from . import readouts, reservoirs
from .model import ESN

__all__ = ["ESN", "readouts", "reservoirs"]
