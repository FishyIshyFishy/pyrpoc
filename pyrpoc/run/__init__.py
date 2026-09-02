"""Executes programs.

May import core/, data/, operations/ and devices/. Pure Python: no Qt, so the
runner is testable with no QApplication. The Qt marshalling the GUI needs lives
in shell/run_bridge.py.
"""

from .program import Program, RunContext
from .runner import RunHandle, Runner, default_program_key
from . import claims

__all__ = ["Program", "RunContext", "Runner", "RunHandle", "default_program_key", "claims"]
