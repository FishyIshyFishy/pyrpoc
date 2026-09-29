"""The ways a program can be started, one runner per file.

A program lists the ones it offers in ``runners``. Each attaches its controls
and callbacks to a ``RunnerContext`` and knows nothing about what hosts it.
"""

from __future__ import annotations

from .arm_and_run import ArmAndRun
from .continuous import Continuous
from .single import Single

__all__ = ["ArmAndRun", "Continuous", "Single"]
