"""``LoopLogOutput``: everything a loop did, in order, written as it happens."""

from __future__ import annotations

from collections.abc import Mapping

from pyrpoc.core.program_output import program_output_registry

from .log import LogOutput
from .loop_events import LOOP_EVENTS, LoopEvent
from .loop_view import LoopFold, LoopView


@program_output_registry.register("loop_log")
class LoopLogOutput(LogOutput[LoopEvent]):
    """A loop's events, from its first, even when it was carried on."""

    def kinds(self) -> Mapping[str, type[LoopEvent]]:
        return LOOP_EVENTS

    def view(self) -> LoopView:
        """What the log says the loop is doing as of its latest event."""
        with self.lock:
            events = list(self.events)
        fold = LoopFold()
        for event in events:
            fold.take(event)
        return fold.view()
