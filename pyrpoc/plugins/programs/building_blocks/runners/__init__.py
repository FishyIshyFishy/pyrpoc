"""The ways a program can be started, one runner per file."""

from __future__ import annotations

from .arm_and_run import ArmAndRun
from .continuous import Continuous
from .single import Single

__all__ = ["ArmAndRun", "Continuous", "Single"]
