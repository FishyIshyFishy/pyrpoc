"""The parameter model: field specs, blocks, and how a set of blocks is held.

A block is a dataclass whose fields carry their spec in ``metadata``, so one
declaration gives the value, default, form row and validation. The block class
is its identity everywhere: declaration, form, workspace and run metadata.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from .data import Library


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
        """The value a run is handed. A field whose value names open data
        fills it in here, so the stored value keeps only the reference."""
        del library
        return value

    def editor(self, parent: Any, context: FieldContext) -> Editor | None:
        """A custom widget, or None for the form's own. Lets a field declared
        outside ``structs`` bring a widget the form cannot know about."""
        del parent, context
        return None


@dataclass
class Editor:
    """One field's widget, as the form drives it. Toolkit-free by type so
    ``structs`` never imports Qt."""

    widget: Any
    get: Callable[[], Any]
    set: Callable[[Any], None]
    # Returns object: the result is ignored, and Qt's connect() returns a handle.
    connect: Callable[[Callable[[], None]], object]
    summary: Callable[[], str]
    spec: Field


@dataclass(frozen=True)
class FieldContext:
    """What the form's owner can offer a field's editor. ``library`` is the
    open data; None for a device configuration, which has none."""

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
    """A dataclass field carrying its spec. Public so field types declared
    outside this module can build their own."""
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
    """Base for parameter blocks. ``label`` is the form's section heading; it
    lives on the class because one block is one section wherever it appears."""

    label: ClassVar[str] = ""


B = TypeVar("B", bound=Group)


def instance_of(value: object, cls: type[B]) -> B:
    """``value`` typed as ``cls``. The stores key every block by its class, so
    this narrows for the type checker rather than validating."""
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


class BlockStore:
    """Every parameter block that exists, one instance per class. Two programs
    declaring the same block share the instance, so an edit under one
    modality is already made under the other."""

    def __init__(self) -> None:
        self._blocks: dict[type, Group] = {}

    def get(self, cls: type[B]) -> B:
        """The instance for ``cls``, created at defaults on first request."""
        if cls not in self._blocks:
            self._blocks[cls] = cls()
        return instance_of(self._blocks[cls], cls)

    def for_program(self, declared: Sequence[type[Group]]) -> BlockMap:
        return BlockMap({cls: self.get(cls) for cls in declared})

    def to_dict(self, only: Sequence[type[Group]] | None = None) -> dict[str, Any]:
        classes = list(only) if only is not None else list(self._blocks)
        return {block_name(cls): encode_block(self.get(cls)) for cls in classes}

    def load_dict(self, raw: Mapping[str, dict[str, Any]], registry: Mapping[str, type]) -> None:
        """Fill the store from a saved state dict. An unknown name is skipped,
        so deleting a block does not strand a workspace; a block that fails
        coercion falls back to defaults, so one bad number costs only itself."""
        for name, values in raw.items():
            cls = registry.get(name)
            if cls is None:
                continue
            try:
                self._blocks[cls] = decode_block(cls, values)
            except ParameterError:
                self._blocks[cls] = cls()

    def validate(self, classes: Sequence[type[Group]]) -> None:
        for cls in classes:
            validate_block(self.get(cls))

    def clear(self) -> None:
        self._blocks.clear()


class BlockMap(Mapping):
    """The blocks one run was given, keyed by class, so ``ctx.params[ScanGroup]``
    is typed as a ``ScanGroup``. An undeclared block is absent."""

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


@dataclass(frozen=True)
class Section:
    label: str
    entries: tuple[tuple[str, Field], ...]


def sections(blocks: Sequence[Group]) -> list[Section]:
    """One section per block, in the order the program declared them."""
    out: list[Section] = []
    for block in blocks:
        entries = tuple((f"{block_name(block)}.{name}", spec) for name, spec in block_fields(block))
        out.append(Section(type(block).label or block_name(block), entries))
    return out


def index(blocks: Sequence[Group]) -> dict[str, Group]:
    """``{"ScanGroup": <instance>}``, what a dotted path resolves against."""
    return {block_name(block): block for block in blocks}


def split_path(path: str) -> tuple[str, str]:
    name, _, attr = path.partition(".")
    if not attr:
        raise KeyError(f"{path!r} is not a <Block>.<field> path")
    return name, attr


def get_path(lookup: Mapping[str, Group], path: str) -> Any:
    name, attr = split_path(path)
    return getattr(lookup[name], attr)


def set_path(lookup: Mapping[str, Group], path: str, value: Any) -> None:
    name, attr = split_path(path)
    setattr(lookup[name], attr, value)


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
    """A copy of ``block`` as a run should see it. Always a copy, so a run
    keeps what it started with while the form edits the shared block, and two
    overlapping runs never share one; field values are immutable, so shallow
    is enough."""
    changes: dict[str, Any] = {}
    for name, spec in block_fields(block):
        value = getattr(block, name)
        resolved = spec.resolve(value, library)
        if resolved is not value:
            changes[name] = resolved
    return replace(block, **changes)
