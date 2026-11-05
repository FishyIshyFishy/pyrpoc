"""``LiveFeed``: one run's live inputs, carried from the thread that sets them to the run's."""

# Program inputs are set on the GUI thread and a run reads on its own, so each change is copied
# under a lock. Nothing here knows who set a value: a panel today, perhaps the run's own logic
# later.

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from pyrpoc.core.program_input import ProgramInput
from pyrpoc.core.recording import utc_now
from pyrpoc.engine.data_library.format import LiveRead

from .program_inputs import ProgramInputsStore


class LiveFeed:
    """The current values of one run's live inputs, following every change until closed."""

    def __init__(
        self,
        inputs: ProgramInputsStore,
        live: Sequence[ProgramInput[Any]],
        on_read: Callable[[LiveRead], None] | None,
    ):
        self.inputs = inputs
        self.on_read = on_read
        self.rules = {each.key: each.rule for each in live}
        self.lock = threading.Lock()
        self.values = {key: inputs.get(key) for key in self.rules}
        # What the run has been given, encoded, so reading an unchanged value logs nothing.
        self.seen = {key: rule.encode(self.values[key]) for key, rule in self.rules.items()}
        inputs.subscribe(self.on_set)

    def keys(self) -> frozenset[str]:
        return frozenset(self.rules)

    def on_set(self, key: str) -> None:
        """Take a new value, on the thread that set it."""
        if key in self.rules:
            with self.lock:
                self.values[key] = self.inputs.get(key)

    def current(self, key: str) -> Any:
        """The value now, logging it if the run has not been given it before."""
        with self.lock:
            value = self.values[key]
        encoded = self.rules[key].encode(value)
        if encoded != self.seen[key]:
            self.seen[key] = encoded
            # None for a run that is not saved: there is no recording to log into.
            if self.on_read is not None:
                self.on_read(LiveRead(utc_now(), key, encoded))
        return value

    def close(self) -> None:
        """Stop following changes, on the thread that sets them."""
        self.inputs.unsubscribe(self.on_set)
