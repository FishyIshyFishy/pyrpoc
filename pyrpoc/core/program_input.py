"""Program inputs: ``ProgramInput`` declarations and the rules their values follow, the values
one run was given, and the ``ProgramInputs`` collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar


class ParameterError(Exception):
    """A parameter value is missing, out of range, or the wrong type."""


@dataclass(frozen=True)
class Field:
    """The rules one value follows, and the label and tooltip it is shown with."""

    label: str
    tooltip: str = ""

    def coerce(self, value: Any) -> Any:
        raise NotImplementedError

    def encode(self, value: Any) -> Any:
        """Value -> JSON-safe. Overridden where the runtime type is not."""
        return value

    def decode(self, raw: Any) -> Any:
        """JSON-safe -> value, raising ``ParameterError`` on bad input."""
        return self.coerce(raw)


T = TypeVar("T")


@dataclass(frozen=True, eq=False)
class ProgramInput(Generic[T]):
    """The declaration of one value a program reads: its key, the rules its
    values follow (which carry its label), its default, whether a session restores it, and
    whether a recording keeps its values."""

    key: str
    rule: Field
    default: T
    # False for a value too large to rewrite on every autosave: each launch starts at the default.
    restored: bool
    # False for a value the programs reading it record in outputs of their own, so a recording's
    # metadata says only when it was taken.
    recorded: bool


# Every declared program input, by key.
program_input_registry: dict[str, ProgramInput[Any]] = {}


def declare(
    key: str, rule: Field, default: T, *, restored: bool = True, recorded: bool = True
) -> ProgramInput[T]:
    """The program input ``key``, which every program that reads it imports from here."""
    # Declared once, so a key can never mean two values: a second program imports the first's.
    if key in program_input_registry:
        raise ValueError(f"program input {key!r} is already declared")
    program_input = ProgramInput(key, rule, default, restored, recorded)
    program_input_registry[key] = program_input
    return program_input


class ProgramInputValues:
    """The value of each program input one run was given."""

    def __init__(self, values: Mapping[str, Any]):
        self._values = dict(values)

    def __getitem__(self, program_input: ProgramInput[T]) -> T:
        if program_input.key not in self._values:
            raise KeyError(
                f"{program_input.key!r} is not an input of this program; it declares "
                f"{sorted(self._values)}"
            )
        return self._values[program_input.key]


class ProgramInputs(Protocol):
    """The current value of every program input, by key."""

    def get(self, key: str) -> Any:
        """The current value, which is the input's default until one is set."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Check ``value`` against the input's rules, store it and tell every
        subscriber. Raises ``ParameterError`` for a value the rules refuse."""
        ...

    def subscribe(self, callback: Callable[[str], None]) -> None:
        """Call ``callback(key)`` after every change."""
        ...

    def unsubscribe(self, callback: Callable[[str], None]) -> None: ...
