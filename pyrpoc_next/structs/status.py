"""Small status enums and the compatibility report shown before a run."""

from __future__ import annotations

from enum import Enum

from attrs import define, field


class ConnectionStatus(Enum):
    """Instrument connection state."""

    untested = "untested"
    ok = "ok"
    failed = "failed"
    disconnected = "disconnected"


class RunStatus(Enum):
    """Acquisition run state."""

    idle = "idle"
    running = "running"
    stopping = "stopping"
    error = "error"


class IssueSeverity(Enum):
    """Whether a compatibility issue blocks a run or only warns."""

    warn = "warn"
    halt = "halt"


@define
class CompatibilityIssue:
    """One problem found while checking a routine before a run."""

    severity: IssueSeverity
    message: str


@define
class CompatibilityReport:
    """The result of a compatibility check: issues to show the user before running."""

    issues: list[CompatibilityIssue] = field(factory=list)

    @property
    def blocked(self) -> bool:
        return any(issue.severity is IssueSeverity.halt for issue in self.issues)

    def add(self, severity: IssueSeverity, message: str) -> None:
        self.issues.append(CompatibilityIssue(severity=severity, message=message))
