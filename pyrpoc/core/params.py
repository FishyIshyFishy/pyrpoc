"""The parameter model: field definitions, shared groups, coercion.

No Qt. The widget half of the old ``backend_utils/parameter_utils.py`` lives in
``shell/param_form.py``; what stays here is label, tooltip, bounds and how a
raw value becomes a real one.

Values live in dataclasses whose fields carry their spec in ``metadata``, so
one declaration serves the value, the default, the form and the validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field, fields, is_dataclass
from typing import Any, Iterable

from .errors import ParameterError
from .modulation import MaskBinding


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


@dataclass(frozen=True)
class MasksField(Field):
    """The Modulation table: mask file, port, line — one row per binding."""

    def coerce(self, value: Any) -> tuple[MaskBinding, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of mask bindings")
        return tuple(MaskBinding.from_dict(row) for row in value)

    def encode(self, value: Any) -> Any:
        return [binding.to_dict() for binding in (value or ())]


# --------------------------------------------------------------------------- #
# Field constructors — each returns a dataclasses.field carrying its spec      #
# --------------------------------------------------------------------------- #


def _spec_field(default: Any, spec: Field, *, factory=None):
    if factory is not None:
        return dc_field(default_factory=factory, metadata={"param": spec})
    return dc_field(default=default, metadata={"param": spec})


def int_field(label, default, *, minimum=None, maximum=None, step=1, tooltip=""):
    return _spec_field(default, IntField(label, tooltip, minimum, maximum, step))


def float_field(label, default, *, minimum=None, maximum=None, step=0.1, decimals=6, tooltip=""):
    return _spec_field(default, FloatField(label, tooltip, minimum, maximum, step, decimals))


def text_field(label, default="", *, tooltip=""):
    return _spec_field(default, TextField(label, tooltip))


def path_field(label, default="", *, dialog_filter="All Files (*)", tooltip=""):
    return _spec_field(default, PathField(label, tooltip, dialog_filter))


def bool_field(label, default=False, *, tooltip=""):
    return _spec_field(default, BoolField(label, tooltip))


def choice_field(label, default, *, choices, tooltip=""):
    return _spec_field(default, ChoiceField(label, tooltip, tuple(choices)))


def channels_field(label, *, num_channels=9, default=None, tooltip=""):
    resolved = tuple(range(num_channels)) if default is None else tuple(default)
    return _spec_field(None, ChannelsField(label, tooltip, num_channels), factory=lambda: resolved)


def masks_field(label="Masks", *, tooltip=""):
    return _spec_field(None, MasksField(label, tooltip), factory=tuple)


def group(cls: type, label: str):
    """A nested parameter group, rendered as its own form section."""
    return dc_field(default_factory=cls, metadata={"group": label, "group_cls": cls})


# --------------------------------------------------------------------------- #
# Group base                                                                   #
# --------------------------------------------------------------------------- #


class Group:
    """Base for parameter groups.

    ``keys`` and ``__getitem__`` are what make ``operation(**p.scan)`` work, so
    operations can stay on loose keyword arguments instead of taking group
    objects.
    """

    def keys(self) -> list[str]:
        return [f.name for f in fields(self) if "param" in f.metadata]

    def __getitem__(self, name: str) -> Any:
        if name not in self.keys():
            raise KeyError(name)
        return getattr(self, name)


# --------------------------------------------------------------------------- #
# Shared groups                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class ScanGroup(Group):
    x_pixels: int = int_field("X Pixels", 512, minimum=8, tooltip="Number of pixels in X")
    y_pixels: int = int_field("Y Pixels", 512, minimum=8, tooltip="Number of pixels in Y")
    extra_left: int = int_field(
        "Extra Steps Left", 300, minimum=0, tooltip="Extra scan steps at the left edge"
    )
    extra_right: int = int_field(
        "Extra Steps Right", 20, minimum=0, tooltip="Extra scan steps at the right edge"
    )
    fast_axis_offset: float = float_field("Fast Axis Offset", 0.0, tooltip="Fast-axis offset")
    fast_axis_amplitude: float = float_field(
        "Fast Axis Amplitude", 1.0, minimum=1e-6, tooltip="Fast-axis amplitude"
    )
    slow_axis_offset: float = float_field("Slow Axis Offset", 0.0, tooltip="Slow-axis offset")
    slow_axis_amplitude: float = float_field(
        "Slow Axis Amplitude", 1.0, minimum=1e-6, tooltip="Slow-axis amplitude"
    )
    dwell_time_us: float = float_field(
        "Dwell Time (us)", 2.0, minimum=0.1, tooltip="Pixel dwell time"
    )

    @property
    def total_x(self) -> int:
        return self.x_pixels + self.extra_left + self.extra_right


@dataclass
class DaqGroup(Group):
    sample_rate_hz: float = float_field(
        "Sample Rate (Hz)",
        100_000.0,
        minimum=1.0,
        maximum=5_000_000.0,
        step=1_000.0,
        tooltip="DAQ sample rate in Hz",
    )


@dataclass
class FlimDaqGroup(DaqGroup):
    """Same field, different default: the FLIM pixel clock divides down from it."""

    sample_rate_hz: float = float_field(
        "Sample Rate (Hz)",
        1_000_000.0,
        minimum=1.0,
        maximum=5_000_000.0,
        step=1_000.0,
        tooltip="DAQ AO sample rate in Hz; the pixel clock is divided down from it",
    )


@dataclass
class ModulationGroup(Group):
    masks: tuple[MaskBinding, ...] = masks_field(
        "Masks", tooltip="Mask files driving digital output lines during the scan"
    )


@dataclass
class SplitGroup(Group):
    t0_samples: int = int_field(
        "t0 Samples", 1, minimum=1, tooltip="Number of samples in the first subpixel window"
    )
    t1_samples: int = int_field(
        "t1 Samples", 0, minimum=0, tooltip="Number of samples to discard between t0 and t2"
    )


@dataclass
class TriggerGroup(Group):
    frame_trigger_pfi: int = int_field(
        "Frame Trigger PFI Line",
        0,
        minimum=0,
        tooltip="PFI line that exports the AO start trigger (frame marker)",
    )
    pixel_clock_ctr: int = int_field(
        "Pixel Clock Counter", 0, minimum=0, tooltip="Counter used to generate the pixel clock"
    )
    pixel_clock_pfi: int = int_field(
        "Pixel Clock PFI Line", 1, minimum=0, tooltip="PFI line that outputs the pixel clock"
    )


@dataclass
class HistogramGroup(Group):
    laser_frequency_mhz: float = float_field(
        "Laser Frequency MHz", 80.0, minimum=0.001, tooltip="Laser repetition rate in MHz"
    )
    histogram_bins: int = int_field(
        "Histogram Bins", 125, minimum=2, tooltip="Number of decay-histogram bins per pixel"
    )
    histogram_binwidth_ps: int = int_field(
        "Histogram Bin Width (ps)",
        100,
        minimum=1,
        tooltip="Bin width in ps (bins x width should span one laser period)",
    )
    frame_settle_s: float = float_field(
        "Frame Settle (s)",
        5e-3,
        minimum=0.0,
        step=1e-3,
        tooltip="Wait after the scan so the last photons reach the measurement",
    )

    @property
    def laser_period_ps(self) -> int:
        return int(round(1e6 / self.laser_frequency_mhz))


# --------------------------------------------------------------------------- #
# Introspection: form sections, dotted paths, serialisation                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Section:
    label: str
    entries: tuple[tuple[str, Field], ...]


def sections(obj_or_cls: Any) -> list[Section]:
    """Ordered form description: nested groups first, then root-level scalars."""
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a parameter dataclass")

    out: list[Section] = []
    root: list[tuple[str, Field]] = []
    for entry in fields(cls):
        if "group" in entry.metadata:
            group_cls = entry.metadata["group_cls"]
            nested = tuple(
                (f"{entry.name}.{inner.name}", inner.metadata["param"])
                for inner in fields(group_cls)
                if "param" in inner.metadata
            )
            out.append(Section(entry.metadata["group"], nested))
        elif "param" in entry.metadata:
            root.append((entry.name, entry.metadata["param"]))
    if root:
        out.append(Section("Acquisition", tuple(root)))
    return out


def spec_at(obj_or_cls: Any, path: str) -> Field:
    for section in sections(obj_or_cls):
        for name, spec in section.entries:
            if name == path:
                return spec
    raise KeyError(path)


def get_path(obj: Any, path: str) -> Any:
    target = obj
    for part in path.split("."):
        target = getattr(target, part)
    return target


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def to_dict(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for entry in fields(obj):
        value = getattr(obj, entry.name)
        if "group" in entry.metadata:
            out[entry.name] = to_dict(value)
        elif "param" in entry.metadata:
            out[entry.name] = entry.metadata["param"].encode(value)
    return out


def from_dict(cls: type, raw: dict[str, Any] | None, *, strict: bool = False) -> Any:
    """Build an instance from a nested plain dict, coercing every value."""
    values = raw or {}
    if not isinstance(values, dict):
        raise ParameterError("parameters must be an object")

    known = {entry.name for entry in fields(cls)}
    if strict:
        unknown = sorted(set(values) - known)
        if unknown:
            raise ParameterError("unknown parameters: " + ", ".join(unknown))

    kwargs: dict[str, Any] = {}
    for entry in fields(cls):
        if entry.name not in values:
            continue
        given = values[entry.name]
        if "group" in entry.metadata:
            kwargs[entry.name] = from_dict(entry.metadata["group_cls"], given, strict=strict)
        elif "param" in entry.metadata:
            kwargs[entry.name] = entry.metadata["param"].decode(given)
    return cls(**kwargs)


def coerce(cls: type, raw: dict[str, Any] | None) -> Any:
    """Strict ``from_dict``: unknown keys are an error."""
    return from_dict(cls, raw, strict=True)


def validate(obj: Any) -> None:
    """Re-run every field's coercion against the values currently held."""
    for section in sections(obj):
        for path, spec in section.entries:
            spec.coerce(get_path(obj, path))
