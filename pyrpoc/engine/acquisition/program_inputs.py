"""``ProgramInputsStore``: the current value of every program input, by key."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pyrpoc.core.program_input import (
    ParameterError,
    ProgramInput,
    ProgramInputValues,
    program_input_registry,
)


class ProgramInputsStore:
    """Every program input's current value, by key: one value per key, shared by
    every program that declares it."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._subscribers: list[Callable[[str], None]] = []

    def get(self, key: str) -> Any:
        """The current value, which is the input's default until one is set."""
        if key not in self._values:
            self._values[key] = program_input_registry[key].default
        return self._values[key]

    def set(self, key: str, value: Any) -> None:
        """Check ``value`` against the input's rules, store it and tell every
        subscriber. Raises ``ParameterError`` for a value the rules refuse."""
        self._values[key] = program_input_registry[key].rule.coerce(value)
        for callback in list(self._subscribers):
            callback(key)

    def subscribe(self, callback: Callable[[str], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def values_for(self, inputs: Sequence[ProgramInput[Any]]) -> ProgramInputValues:
        """The current values of ``inputs``, which a run keeps however they change later."""
        return ProgramInputValues({each.key: self.get(each.key) for each in inputs})

    def encode(self, inputs: Sequence[ProgramInput[Any]]) -> dict[str, Any]:
        """The current values of ``inputs`` as JSON, by key."""
        return {each.key: each.rule.encode(self.get(each.key)) for each in inputs}

    def to_dict(self) -> dict[str, Any]:
        """Every restored value set or read so far, as JSON, by key."""
        return {
            key: program_input.rule.encode(value)
            for key, value in self._values.items()
            if (program_input := program_input_registry[key]).restored
        }

    def load_dict(self, raw: Mapping[str, Any]) -> None:
        """Take values from saved JSON, skipping any whose input no longer exists,
        is not restored, or whose value its rules refuse."""
        for key, value in raw.items():
            # A session saved before an input was declared restored=False may still hold it.
            if key not in program_input_registry or not program_input_registry[key].restored:
                continue
            try:
                self._values[key] = program_input_registry[key].rule.decode(value)
            except ParameterError:
                continue

    def clear(self) -> None:
        """Forget every value, so each input is back at its default."""
        self._values.clear()
