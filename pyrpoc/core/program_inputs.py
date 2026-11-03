"""Program inputs: the rules values follow, ``ProgramInput`` declarations, the
values one run was given, and the ``ProgramInputs`` protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.core.library import Library


class ParameterError(Exception):
    """A parameter value is missing, out of range, or the wrong type."""


@dataclass(frozen=True)
class Field:
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

    def resolve(self, value: Any, library: Library) -> Any:
        """The value a run is handed: a field whose value names open data fills
        it in here."""
        del library
        return value

    def editor(self, parent: Any, context: FieldContext) -> Editor | None:
        """A custom widget for this field, or None for the form's default."""
        del parent, context
        return None


@dataclass
class Editor:
    """One field's widget, as the form drives it."""

    # Typed as Any so core never imports Qt.
    widget: Any
    get: Callable[[], Any]
    set: Callable[[Any], None]
    # Returns object: the result is ignored, and Qt's connect() returns a handle.
    connect: Callable[[Callable[[], None]], object]
    summary: Callable[[], str]
    spec: Field


@dataclass(frozen=True)
class FieldContext:
    """What a field's editor can draw on: the open data, or None for a device
    configuration."""

    library: Library | None = None


@dataclass(frozen=True)
class IntField(Field):
    minimum: int | None = None
    maximum: int | None = None
    step: int = 1

    def coerce(self, value: Any) -> int:
        if isinstance(value, bool):
            raise ParameterError(f"{self.label}: expected an integer, got a boolean")
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"{self.label}: expected an integer") from exc
        if self.minimum is not None and out < self.minimum:
            raise ParameterError(f"{self.label}: must be >= {self.minimum}")
        if self.maximum is not None and out > self.maximum:
            raise ParameterError(f"{self.label}: must be <= {self.maximum}")
        return out


@dataclass(frozen=True)
class FloatField(Field):
    minimum: float | None = None
    maximum: float | None = None
    step: float = 0.1
    decimals: int = 6

    def coerce(self, value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"{self.label}: expected a number") from exc
        if self.minimum is not None and out < self.minimum:
            raise ParameterError(f"{self.label}: must be >= {self.minimum}")
        if self.maximum is not None and out > self.maximum:
            raise ParameterError(f"{self.label}: must be <= {self.maximum}")
        return out


@dataclass(frozen=True)
class TextField(Field):
    def coerce(self, value: Any) -> str:
        return "" if value is None else str(value)


@dataclass(frozen=True)
class PathField(Field):
    dialog_filter: str = "All Files (*)"

    def coerce(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value)
        if not text.strip():
            return ""
        if text.rstrip().endswith(("\\", "/")):
            raise ParameterError(f"{self.label}: path must include a filename")
        return text


@dataclass(frozen=True)
class BoolField(Field):
    def coerce(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise ParameterError(f"{self.label}: expected true or false")


@dataclass(frozen=True)
class ChoiceField(Field):
    choices: tuple[str, ...] = ()

    def coerce(self, value: Any) -> str:
        text = str(value)
        if text not in self.choices:
            raise ParameterError(f"{self.label}: must be one of {list(self.choices)}")
        return text


@dataclass(frozen=True)
class ChannelsField(Field):
    """A row of toggleable channel buttons. The value is a sorted int tuple."""

    num_channels: int = 9

    def coerce(self, value: Any) -> tuple[int, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of channel indices")
        try:
            out = sorted({int(item) for item in value})
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"{self.label}: channel indices must be integers") from exc
        for index in out:
            if index < 0 or index >= self.num_channels:
                raise ParameterError(
                    f"{self.label}: channel {index} is outside 0..{self.num_channels - 1}"
                )
        return tuple(out)

    def encode(self, value: Any) -> Any:
        return list(value)


T = TypeVar("T")


@dataclass(frozen=True, eq=False)
class ProgramInput(Generic[T]):
    """The declaration of one value a program reads: its key, the rules its
    values follow (which carry its label), and its default."""

    key: str
    rule: Field
    default: T


# Every declared program input, by key.
program_input_registry: dict[str, ProgramInput[Any]] = {}


def declare(key: str, rule: Field, default: T) -> ProgramInput[T]:
    """The program input ``key``, shared by every program that declares it."""
    existing = program_input_registry.get(key)
    if existing is not None:
        # The same key must mean the same value everywhere it is declared.
        if type(existing.rule) is not type(rule):
            raise TypeError(
                f"program input {key!r} is declared as both {type(existing.rule).__name__} "
                f"and {type(rule).__name__}"
            )
        return existing
    program_input = ProgramInput(key, rule, default)
    program_input_registry[key] = program_input
    return program_input


def int_input(
    key, label, default, *, minimum=None, maximum=None, step=1, tooltip=""
) -> ProgramInput[int]:
    return declare(key, IntField(label, tooltip, minimum, maximum, step), default)


def float_input(
    key, label, default, *, minimum=None, maximum=None, step=0.1, decimals=6, tooltip=""
) -> ProgramInput[float]:
    return declare(key, FloatField(label, tooltip, minimum, maximum, step, decimals), default)


def bool_input(key, label, default, *, tooltip="") -> ProgramInput[bool]:
    return declare(key, BoolField(label, tooltip), default)


def choice_input(key, label, default, *, choices, tooltip="") -> ProgramInput[str]:
    return declare(key, ChoiceField(label, tooltip, tuple(choices)), default)


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


def values_from_parameters(
    inputs: Sequence[ProgramInput[Any]], parameters: Mapping[str, Any]
) -> ProgramInputValues:
    """``inputs`` as a recording's ``parameters`` hold them, which must name every one."""
    return ProgramInputValues({each.key: each.rule.decode(parameters[each.key]) for each in inputs})


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
