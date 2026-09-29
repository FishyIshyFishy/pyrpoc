"""The parameter model: field definitions, blocks, coercion.

No Qt, no hardware, no instrument vocabulary. The widget half lives in
``shell/param_form.py``; the blocks themselves live with the code that declares
them. What stays here is label, tooltip, bounds, how a raw value becomes a real
one, and how a set of blocks is held, addressed and serialised.

A **block** is a dataclass whose fields carry their spec in ``metadata``, so one
declaration serves the value, the default, the form and the validation. A
program declares which block classes it wants; the block class is the identity
everywhere -- declaration key, form key, session key, metadata key.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field, fields, is_dataclass
from typing import Any, ClassVar, Iterable, Iterator, Mapping, Sequence, TypeVar

from .errors import ParameterError


# --------------------------------------------------------------------------- #
# Field specs                                                                  #
# --------------------------------------------------------------------------- #


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
        """JSON-safe -> value, with validation."""
        return self.coerce(raw)


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
        return list(value or ())


# --------------------------------------------------------------------------- #
# Field constructors — each returns a dataclasses.field carrying its spec      #
# --------------------------------------------------------------------------- #


def spec_field(default: Any, spec: Field, *, factory=None):
    """A dataclass field carrying its parameter spec. Public so blocks declared
    outside this module can define their own field types."""
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


# --------------------------------------------------------------------------- #
# Block base                                                                   #
# --------------------------------------------------------------------------- #


class Group:
    """Base for parameter blocks.

    Two jobs, both real. ``label`` is the section heading the form draws, which
    is why it lives on the class rather than at each declaration site: one block
    is one section wherever it appears.

    It is deliberately not a mapping. It used to carry ``keys`` and
    ``__getitem__`` so ``f(**p.some_group)`` would work; every caller now takes
    the block itself, so a renamed field is a type error instead of a
    ``TypeError`` on the first frame.
    """

    label: ClassVar[str] = ""


B = TypeVar("B", bound=Group)


def block_fields(block_or_cls: Any) -> list[tuple[str, Field]]:
    """``(name, spec)`` for every parameter field on a block."""
    cls = block_or_cls if isinstance(block_or_cls, type) else type(block_or_cls)
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a parameter block")
    return [(f.name, f.metadata["param"]) for f in fields(cls) if "param" in f.metadata]


def block_name(block_or_cls: Any) -> str:
    """The serialisation and addressing key for a block: its class name."""
    cls = block_or_cls if isinstance(block_or_cls, type) else type(block_or_cls)
    return cls.__name__


# --------------------------------------------------------------------------- #
# Holding blocks: the store and the run-time map                               #
# --------------------------------------------------------------------------- #


class BlockStore:
    """Every parameter block that exists, one instance per class.

    This is what makes a block shared: two programs declaring ``ScanGroup`` are
    handed the same object, so editing it in one modality's form is editing it
    in the other's. The session file is this store, encoded.
    """

    def __init__(self) -> None:
        self._blocks: dict[type, Group] = {}

    def get(self, cls: type[B]) -> B:
        """The instance for ``cls``, created at defaults on first request."""
        block = self._blocks.get(cls)
        if block is None:
            block = cls()
            self._blocks[cls] = block
        return block  # type: ignore[return-value]

    def has(self, cls: type) -> bool:
        return cls in self._blocks

    def for_program(self, declared: Sequence[type[Group]]) -> "BlockMap":
        return BlockMap({cls: self.get(cls) for cls in declared})

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self, only: Sequence[type[Group]] | None = None) -> dict[str, Any]:
        classes = list(only) if only is not None else list(self._blocks)
        return {block_name(cls): encode_block(self.get(cls)) for cls in classes}

    def load_dict(self, raw: Mapping[str, Any] | None, registry: Mapping[str, type]) -> None:
        """Fill the store from a saved state dict.

        A name with no class in ``registry`` is skipped rather than fatal: a
        block can be deleted from the source without stranding a session file.
        A block whose values fail coercion falls back to its defaults, so one
        bad number cannot cost the whole rig its settings.
        """
        for name, values in (raw or {}).items():
            cls = registry.get(str(name))
            if cls is None or not isinstance(values, dict):
                continue
            try:
                self._blocks[cls] = decode_block(cls, values)
            except Exception:
                self._blocks[cls] = cls()

    def validate(self, only: Sequence[type[Group]] | None = None) -> None:
        classes = list(only) if only is not None else list(self._blocks)
        for cls in classes:
            validate_block(self.get(cls))

    def clear(self) -> None:
        self._blocks.clear()


class BlockMap(Mapping):
    """The blocks one run was given, keyed by class.

    The parameter twin of ``DeviceMap``: ``ctx.params[ScanGroup]`` is typed as a
    ``ScanGroup`` for the same reason ``ctx.devices[DAQ]`` is typed as a ``DAQ``.
    A block the program did not declare is absent, which is the containment
    ``uses`` already gives devices.
    """

    def __init__(self, blocks: Mapping[type, Group] | None = None):
        self._blocks: dict[type, Group] = dict(blocks or {})

    def __getitem__(self, key: type[B]) -> B:
        try:
            return self._blocks[key]  # type: ignore[return-value]
        except KeyError:
            raise KeyError(
                f"{getattr(key, '__name__', key)!r} is not in this program's params; "
                f"it declares {sorted(block_name(c) for c in self._blocks)}"
            ) from None

    def __iter__(self) -> Iterator[type]:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BlockMap({sorted(block_name(c) for c in self._blocks)})"


# --------------------------------------------------------------------------- #
# Form description and dotted addressing                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Section:
    label: str
    entries: tuple[tuple[str, Field], ...]


def sections(blocks: Sequence[Group]) -> list[Section]:
    """One section per block, in the order the program declared them."""
    out: list[Section] = []
    for block in blocks:
        entries = tuple(
            (f"{block_name(block)}.{name}", spec) for name, spec in block_fields(block)
        )
        out.append(Section(type(block).label or block_name(block), entries))
    return out


def index(blocks: Sequence[Group]) -> dict[str, Group]:
    """``{"ScanGroup": <instance>}`` -- what a dotted path resolves against."""
    return {block_name(block): block for block in blocks}


def split_path(path: str) -> tuple[str, str]:
    name, _, attr = path.partition(".")
    if not attr:
        raise KeyError(f"{path!r} is not a <Block>.<field> path")
    return name, attr


def get_path(blocks: Sequence[Group] | Mapping[str, Group], path: str) -> Any:
    lookup = blocks if isinstance(blocks, Mapping) else index(blocks)
    name, attr = split_path(path)
    return getattr(lookup[name], attr)


def set_path(blocks: Sequence[Group] | Mapping[str, Group], path: str, value: Any) -> None:
    lookup = blocks if isinstance(blocks, Mapping) else index(blocks)
    name, attr = split_path(path)
    setattr(lookup[name], attr, value)


def spec_at(blocks: Sequence[Group] | Mapping[str, Group], path: str) -> Field:
    lookup = blocks if isinstance(blocks, Mapping) else index(blocks)
    name, attr = split_path(path)
    for field_name, spec in block_fields(lookup[name]):
        if field_name == attr:
            return spec
    raise KeyError(path)


# --------------------------------------------------------------------------- #
# One block in and out of plain data                                           #
# --------------------------------------------------------------------------- #


def encode_block(block: Any) -> dict[str, Any]:
    return {name: spec.encode(getattr(block, name)) for name, spec in block_fields(block)}


def decode_block(cls: type, raw: Mapping[str, Any] | None, *, strict: bool = False) -> Any:
    """Build a block from a plain dict, coercing every value."""
    values = raw or {}
    if not isinstance(values, Mapping):
        raise ParameterError("parameters must be an object")

    known = {name for name, _ in block_fields(cls)}
    if strict:
        unknown = sorted(set(values) - known)
        if unknown:
            raise ParameterError("unknown parameters: " + ", ".join(unknown))

    kwargs = {
        name: spec.decode(values[name])
        for name, spec in block_fields(cls)
        if name in values
    }
    return cls(**kwargs)


def validate_block(block: Any) -> None:
    """Re-run every field's coercion against the values currently held."""
    for name, spec in block_fields(block):
        spec.coerce(getattr(block, name))


#: Device configurations are blocks too -- same fields, same form, same
#: encoding -- but they are per-instance rather than per-class and persist with
#: their device, so they never enter the BlockStore. These aliases are what
#: ``devices/base.py`` calls.
to_dict = encode_block
from_dict = decode_block
validate = validate_block


def coerce(cls: type, raw: Mapping[str, Any] | None) -> Any:
    """Strict ``decode_block``: unknown keys are an error."""
    return decode_block(cls, raw, strict=True)
