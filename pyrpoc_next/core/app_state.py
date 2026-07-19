"""The live runtime state the core coordinates.

Holds the actual objects — instruments, display sinks, the current routine — not ids.
Displays enter as a DisplaySink protocol, so nothing here imports Qt.
"""

from __future__ import annotations

from attrs import define, field

from pyrpoc_next.structs.routine import Routine
from pyrpoc_next.structs.status import RunStatus


@define
class AppState:
    """Everything the running app holds in memory."""

    instruments: list = field(factory=list)
    displays: list = field(factory=list)
    routine: Routine = field(factory=Routine)
    run_status: RunStatus = field(default=RunStatus.idle)

    def instrument_for(self, key):
        """Return the first added instrument with this key, or None."""
        return next((instrument for instrument in self.instruments if instrument.key is key), None)
