"""Configuration persistence only.

What exists, how it is configured, and the layout -- so the workbench comes back
on relaunch. May import core/. No Qt: the file path is supplied by the caller
rather than looked up, which keeps this testable headless.
"""

from .state import SCHEMA_VERSION, DeviceState, SessionState, ViewState
from .store import SessionStore, default_session_path

__all__ = [
    "SCHEMA_VERSION",
    "SessionState",
    "DeviceState",
    "ViewState",
    "SessionStore",
    "default_session_path",
]
