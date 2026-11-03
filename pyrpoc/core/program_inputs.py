"""Program parameters: field specs, the blocks (``Group``) that hold them, and
``BlockStore``, which holds one instance of every block."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, replace
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

from pyrpoc.core.registry import Registry

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


def spec_field(default: Any, spec: Field, *, factory=None):
    """A dataclass field carrying its spec."""
    if factory is not None:
        return dc_field(default_factory=factory, metadata={"param": spec})
    return dc_field(default=default, metadata={"param": spec})


def int_field(label, default, *, minimum=None, maximum=None, step=1, tooltip=""):
    return spec_field(default, IntField(label, tooltip, minimum, maximum, step))


def float_field(label, default, *, minimum=None, maximum=None, step=0.1, decimals=6, tooltip=""):
    return spec_field(default, FloatField(label, tooltip, minimum, maximum, step, decimals))


def text_field(label, default="", *, tooltip=""):
    return spec_field(default, TextField(label, tooltip))


def path_field(label, default="", *, dialog_filter="All Files (*)", tooltip=""):
    return spec_field(default, PathField(label, tooltip, dialog_filter))


def bool_field(label, default=False, *, tooltip=""):
    return spec_field(default, BoolField(label, tooltip))


def choice_field(label, default, *, choices, tooltip=""):
    return spec_field(default, ChoiceField(label, tooltip, tuple(choices)))


def channels_field(label, *, num_channels=9, default=None, tooltip=""):
    resolved = tuple(range(num_channels)) if default is None else tuple(default)
    return spec_field(None, ChannelsField(label, tooltip, num_channels), factory=lambda: resolved)


# A dataclass with no fields, so every block type-checks as a dataclass
# instance and ``dataclasses.replace`` accepts it.
@dataclass
class Group:
    """Base for parameter blocks: a dataclass whose fields carry their spec.
    ``label`` is the block's section heading."""

    label: ClassVar[str] = ""


B = TypeVar("B", bound=Group)


def instance_of(value: object, cls: type[B]) -> B:
    """``value`` typed as ``cls``, raising if it is not one."""
    if not isinstance(value, cls):
        raise TypeError(f"{value!r} is stored under {cls.__name__} but is not one")
    return value


def block_fields(block_or_cls: Any) -> list[tuple[str, Field]]:
    """``(name, spec)`` for every parameter field on a block."""
    return [(f.name, f.metadata["param"]) for f in fields(block_or_cls) if "param" in f.metadata]


def block_name(block_or_cls: Any) -> str:
    """The serialisation and addressing key for a block: its class name."""
    cls = block_or_cls if isinstance(block_or_cls, type) else type(block_or_cls)
    return cls.__name__


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


class BlockMap(Mapping):
    """The blocks one run was given, keyed by class."""

    def __init__(self, blocks: Mapping[type, Group]):
        self._blocks: dict[type, Group] = dict(blocks)

    def __getitem__(self, key: type[B]) -> B:
        if key not in self._blocks:
            raise KeyError(
                f"{key.__name__!r} is not in this program's params; "
                f"it declares {sorted(block_name(c) for c in self._blocks)}"
            )
        return instance_of(self._blocks[key], key)

    def __iter__(self) -> Iterator[type]:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BlockMap({sorted(block_name(c) for c in self._blocks)})"


def split_path(path: str) -> tuple[str, str]:
    """``"ScanGroup.dwell_time_us"`` as its block name and field name."""
    name, _, attr = path.partition(".")
    if not attr:
        raise KeyError(f"{path!r} is not a <Block>.<field> path")
    return name, attr


def encode_block(block: Any) -> dict[str, Any]:
    return {name: spec.encode(getattr(block, name)) for name, spec in block_fields(block)}


def decode_block(cls: type, raw: Mapping[str, Any]) -> Any:
    """Build a block from a plain dict, coercing every value. Missing keys
    take the field's default."""
    kwargs = {name: spec.decode(raw[name]) for name, spec in block_fields(cls) if name in raw}
    return cls(**kwargs)


def validate_block(block: Any) -> None:
    """Re-run every field's coercion against the values currently held."""
    for name, spec in block_fields(block):
        spec.coerce(getattr(block, name))


def resolve_block(block: B, library: Library) -> B:
    """A copy of ``block`` as a run should see it."""
    # Always a copy, so a run keeps what it started with while the form edits the
    # shared block; field values are immutable, so shallow is enough.
    changes: dict[str, Any] = {}
    for name, spec in block_fields(block):
        value = getattr(block, name)
        resolved = spec.resolve(value, library)
        if resolved is not value:
            changes[name] = resolved
    return replace(block, **changes)


# Keyed by class name, a block's identity in the form, session and metadata.
block_registry: Registry[Group] = Registry("BlockRegistry", Group, stamp=None)

R = TypeVar("R", bound=type)


def block(cls: R) -> R:
    """Register a parameter block under its class name."""
    return block_registry.register(cls.__name__)(cls)
