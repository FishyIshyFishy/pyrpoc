"""The ways a program can be started, one runner per file."""

from __future__ import annotations

from .continuous import Continuous
from .run_on_pick import RunOnPick
from .single import Single

__all__ = ["Continuous", "RunOnPick", "Single"]
