"""``LiveFeed``: one run's live inputs, carried from the thread that sets them to the run's."""

# Program inputs are set on the GUI thread and a run reads on its own, so each change is copied
# under a lock. Nothing here knows who set a value: a panel today, perhaps the run's own logic
# later.

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from pyrpoc.core.program_input import ProgramInput
from pyrpoc.core.program_output import utc_now
from pyrpoc.engine.data_library.format import LiveRead

from .program_inputs import ProgramInputsStore


class LiveFeed:
    """The current values of one run's live inputs, following every change until closed."""

    def __init__(
        self,
        inputs: ProgramInputsStore,
        live: Sequence[ProgramInput[Any]],
        run_id: int,
        on_read: Callable[[LiveRead], None] | None,
    ):
        self.inputs = inputs
        self.run_id = run_id
        self.on_read = on_read
        self.declared = {each.key: each for each in live}
        self.lock = threading.Lock()
        self.values = {key: inputs.get(key) for key in self.declared}
        # What the run has been given, so reading an unchanged value logs nothing.
        self.seen = {key: self.mark(key, value) for key, value in self.values.items()}
        inputs.subscribe(self.on_set)

    def keys(self) -> frozenset[str]:
        return frozenset(self.declared)

    def mark(self, key: str, value: Any) -> Any:
        """What tells one value of ``key`` from another: its encoding, or for an input that
        is not persisted the value itself, since every set stores a new one."""
        # Encoding a value too large to persist on every read is the cost this avoids.
        program_input = self.declared[key]
        return program_input.rule.encode(value) if program_input.persisted else value

    def on_set(self, key: str) -> None:
        """Take a new value, on the thread that set it."""
        if key in self.declared:
            with self.lock:
                self.values[key] = self.inputs.get(key)

    def current(self, key: str) -> Any:
        """The value now, logging it if the run has not been given it before."""
        with self.lock:
            value = self.values[key]
        mark = self.mark(key, value)
        if not self.changed(key, mark):
            return value
        self.seen[key] = mark
        # None for a run that is not saved: there is no recording to log into.
        if self.on_read is not None:
            persisted = self.declared[key].persisted
            encoded = mark if persisted else None
            self.on_read(LiveRead(self.run_id, utc_now(), key, encoded, persisted))
        return value

    def changed(self, key: str, mark: Any) -> bool:
        if self.declared[key].persisted:
            return mark != self.seen[key]
        return mark is not self.seen[key]

    def close(self) -> None:
        """Stop following changes, on the thread that sets them."""
        self.inputs.unsubscribe(self.on_set)
