"""Program inputs: ``ProgramInput`` declarations and the rules their values follow, the values
one run was given, the ``ProgramInputs`` collection, and ``InputSection``, how a panel lays
them out."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar


class ParameterError(Exception):
    """A parameter value is missing, out of range, or the wrong type."""


@dataclass(frozen=True)
class Field:
    label: str
    tooltip: str = ""
    # False for a value another input's editor already shows, so a form leaves it out.
    shown: bool = field(default=True, kw_only=True)

    def coerce(self, value: Any) -> Any:
        raise NotImplementedError

    def encode(self, value: Any) -> Any:
        """Value -> JSON-safe. Overridden where the runtime type is not."""
        return value

    def decode(self, raw: Any) -> Any:
        """JSON-safe -> value, raising ``ParameterError`` on bad input."""
        return self.coerce(raw)

    def editor(self, parent: Any, values: FieldValues) -> Editor | None:
        """A custom widget for this field, or None for the form's default. ``values`` is the
        form's, for an editor that shows other fields' values beside its own."""
        del parent, values
        return None


class FieldValues(Protocol):
    """The values a form shows, by key."""

    def get(self, key: str) -> Any: ...


@dataclass
class Editor:
    """One field's widget, and how a form reads, writes and watches it."""

    # Typed as Any so core never imports Qt.
    widget: Any
    get: Callable[[], Any]
    set: Callable[[Any], None]
    # Returns object: the result is ignored, and Qt's connect() returns a handle.
    connect: Callable[[Callable[[], None]], object]
    summary: Callable[[], str]
    spec: Field
    # Whether a form gives it a whole row rather than one cell of a shared row.
    wide: bool = False


T = TypeVar("T")


@dataclass(frozen=True, eq=False)
class ProgramInput(Generic[T]):
    """The declaration of one value a program reads: its key, the rules its
    values follow (which carry its label), its default, and whether it is written to files."""

    key: str
    rule: Field
    default: T
    # False for a value too large to write into JSON: the session and recordings leave it out,
    # so it starts at its default each launch.
    persisted: bool


# Every declared program input, by key.
program_input_registry: dict[str, ProgramInput[Any]] = {}


def declare(key: str, rule: Field, default: T, *, persisted: bool = True) -> ProgramInput[T]:
    """The program input ``key``, which every program that reads it imports from here."""
    # Declared once, so a key can never mean two values: a second program imports the first's.
    if key in program_input_registry:
        raise ValueError(f"program input {key!r} is already declared")
    program_input = ProgramInput(key, rule, default, persisted)
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


@dataclass(frozen=True)
class InputSection:
    """A heading and the program inputs under it: layout only, written by each program."""

    label: str
    inputs: tuple[ProgramInput[Any], ...]
