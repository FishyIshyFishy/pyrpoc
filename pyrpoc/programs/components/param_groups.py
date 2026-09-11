"""Every parameter block a program can be built from.

A block is a reusable unit of configuration. A program declares the blocks it
wants and the store hands it the one instance of each, so two modalities
declaring ``ScanGroup`` are configuring the same scan -- change the geometry in
confocal and FLIM already has it.

These live here rather than in ``core/`` because they are what this instrument
is, not what the software is. ``core/params.py`` holds the machinery; this holds
the content.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, ClassVar, Iterable, Sequence, TypeVar

import numpy as np

from pyrpoc.core.errors import ParameterError
from pyrpoc.core.params import (
    Field,
    Group,
    bool_field,
    channels_field,
    choice_field,
    float_field,
    int_field,
    spec_field,
    text_field,
)

#: name -> class, for turning a saved state dict back into blocks.
BLOCKS: dict[str, type[Group]] = {}

#: The decorated class itself, so ``@block`` returns ``type[ScanGroup]`` rather
#: than ``type[Group]``. Without it every registered block is just a ``Group``
#: downstream and ``scan.x_pixels`` type-checks against nothing -- the same
#: trap ``core/registry.py`` documents for devices, views and programs.
B = TypeVar("B", bound=type)


def block(cls: B) -> B:
    """Register a block under its class name, which is its identity everywhere.

    A decorator rather than an explicit-key registry because the key *is* the
    class name -- writing it twice is one more thing to get out of step.
    """
    if not issubclass(cls, Group):
        raise TypeError(f"{cls.__name__} must inherit from Group")
    if cls.__name__ in BLOCKS:
        raise KeyError(f"{cls.__name__!r} is already registered as a block")
    BLOCKS[cls.__name__] = cls
    return cls


# --------------------------------------------------------------------------- #
# Masks: an authored region plus the digital line it drives                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Mask:
    """One authored mask wired to one digital output line.

    The same two halves as ``Point``, for the same reason. ``array`` is what the
    hardware does, so a run needs nothing but its parameters to proceed;
    ``source_id`` and ``source_label`` are provenance on top of that, saying
    which library entry this came from. The label is stored rather than resolved
    because a dataset id means nothing to anyone reading the metadata six months
    later, and the entry it named may not be open any more.

    The array is carried rather than a reference to the library because a
    program has no library: ``RunContext`` hands out this run's own parameters
    and datasets and nothing else, and widening that to every open dataset in
    order to fetch a mask would be a much larger hole than this feature is
    worth. So it is resolved once, when the user picks it -- which is exactly
    when ``Point`` resolves a clicked pixel into volts.

    ``array`` is ``compare=False`` because this is a frozen dataclass: the
    generated ``__eq__`` would compare two arrays elementwise and then call
    ``bool()`` on the result, which raises.
    """

    source_id: str = ""
    source_label: str = ""
    array: np.ndarray | None = dc_field(default=None, compare=False)
    port: int = 0
    line: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "source_label", str(self.source_label))
        object.__setattr__(self, "port", int(self.port))
        object.__setattr__(self, "line", int(self.line))
        if self.array is not None:
            array = np.asarray(self.array)
            if array.ndim != 2:
                raise ParameterError(f"a mask must be 2D, got shape={array.shape}")
            object.__setattr__(self, "array", array)

    @property
    def resolved(self) -> bool:
        """Whether this mask still carries the pixels it names."""
        return self.array is not None

    def describe(self) -> str:
        """Which library entry this is, for a row that has lost it."""
        return self.source_label or self.source_id or "no mask"

    def channel(self, device_name: str) -> str:
        """The NI-DAQ channel string this mask drives."""
        return f"{device_name}/port{self.port}/line{self.line}"

    def to_dict(self) -> dict[str, Any]:
        """Provenance only. The array is data, and this is a parameter."""
        return {
            "source_id": self.source_id,
            "source_label": self.source_label,
            "port": self.port,
            "line": self.line,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "Mask":
        if isinstance(raw, Mask):
            return raw
        if not isinstance(raw, dict):
            raise ParameterError("a mask must be an object with source_id/port/line")
        return cls(
            source_id=str(raw.get("source_id", "")),
            source_label=str(raw.get("source_label", "")),
            port=int(raw.get("port", 0)),
            line=int(raw.get("line", 0)),
        )


@dataclass(frozen=True)
class MasksField(Field):
    """The Modulation table: library entry, port, line — one row per mask."""

    def coerce(self, value: Any) -> tuple[Mask, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of masks")
        return tuple(Mask.from_dict(row) for row in value)

    def encode(self, value: Any) -> Any:
        return [mask.to_dict() for mask in (value or ())]

    def decode(self, raw: Any) -> tuple[Mask, ...]:
        """Nothing. Masks do not survive a relaunch.

        The only field in the application that overrides ``decode``, because it
        is the only one whose value is not self-contained. ``encode`` is shared
        by the run metadata and the session file, and the two want different
        things from it: a run must record exactly which masks drove which lines,
        while a session reload cannot honour that record at all -- the library
        is empty at launch, so every id in it is dangling.

        Returning the rows without their arrays would put the modality one
        silent step from acquiring with a mask that is not there. Returning
        nothing is the honest answer until the library itself persists, at which
        point this override is what should go away.
        """
        del raw
        return ()


def masks_field(label="Masks", *, tooltip=""):
    return spec_field(None, MasksField(label, tooltip), factory=tuple)


# --------------------------------------------------------------------------- #
# Points: where the galvos park, and which pixel said so                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Point:
    """One parked galvo position.

    ``fast_v``/``slow_v`` are what the hardware does, so a run reproduces from
    its saved parameters alone with no reference to the image it came from.
    ``source_id`` and the pixel are provenance on top of that: they say this
    spectrum is pixel (37, 204) of a particular run, which is worth the fields.
    A point typed by hand has no source, which is what the defaults mean.

    ``source_label`` is the human half of that identity, and it is stored rather
    than resolved for the same reason ``Provenance.name`` is: a dataset id means
    nothing to anyone reading the metadata six months later, and the dataset it
    named may not be open any more.

    Voltages rather than pixels are the parameter because a pixel is only
    meaningful against a scan geometry, and that geometry belongs to the image
    -- not to the program being configured.
    """

    fast_v: float = 0.0
    slow_v: float = 0.0
    source_id: str = ""
    source_label: str = ""
    pixel_x: int = -1
    pixel_y: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "fast_v", float(self.fast_v))
        object.__setattr__(self, "slow_v", float(self.slow_v))
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "source_label", str(self.source_label))
        object.__setattr__(self, "pixel_x", int(self.pixel_x))
        object.__setattr__(self, "pixel_y", int(self.pixel_y))

    @property
    def picked(self) -> bool:
        """Whether this point came from a display rather than the keyboard."""
        return bool(self.source_id) and self.pixel_x >= 0 and self.pixel_y >= 0

    def describe(self) -> str:
        """Where this point came from, for the label under the spin boxes."""
        if not self.picked:
            return "typed"
        return f"{self.source_label or self.source_id} px ({self.pixel_x}, {self.pixel_y})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast_v": self.fast_v,
            "slow_v": self.slow_v,
            "source_id": self.source_id,
            "source_label": self.source_label,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "Point":
        if isinstance(raw, Point):
            return raw
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ParameterError("a point must be an object with fast_v/slow_v")
        return cls(
            fast_v=float(raw.get("fast_v", 0.0)),
            slow_v=float(raw.get("slow_v", 0.0)),
            source_id=str(raw.get("source_id", "")),
            source_label=str(raw.get("source_label", "")),
            pixel_x=int(raw.get("pixel_x", -1)),
            pixel_y=int(raw.get("pixel_y", -1)),
        )


@dataclass(frozen=True)
class PointField(Field):
    """A galvo position, with a widget that can pick one off a display.

    ``coerce`` has to be idempotent on a live ``Point``: the form coerces what
    its widget hands back, and ``Runner.start`` re-coerces every held value
    through ``BlockStore.validate`` before a run begins. ``decode`` is inherited
    -- ``Field.decode`` is ``coerce``, and ``from_dict`` takes both forms.
    """

    def coerce(self, value: Any) -> Point:
        return Point.from_dict(value)

    def encode(self, value: Any) -> Any:
        return Point.from_dict(value).to_dict()


def point_field(label="Target", *, tooltip=""):
    return spec_field(None, PointField(label, tooltip), factory=Point)


# --------------------------------------------------------------------------- #
# Shared across modalities                                                     #
# --------------------------------------------------------------------------- #


@block
@dataclass
class ScanGroup(Group):
    """How the beam is scanned, and how many times.

    ``num_frames`` lives here rather than in a block of its own because it
    decides how the imaging is done, the same as the geometry and the dwell
    time. Nothing outside a program reads it: a program with no frame concept
    declares a different block, and then nothing anywhere has to supply a count
    it does not have.
    """

    label: ClassVar[str] = "Scan"

    num_frames: int = int_field(
        "Frames", 1, minimum=1, tooltip="Number of frames to capture"
    )
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

    def voltage_at(self, x: int, y: int) -> tuple[float, float]:
        """The (fast, slow) volts at which displayed pixel ``(x, y)`` was sampled.

        The inverse of ``generate_raster_waveform``'s axis construction, and it
        lives here because this block is what the geometry *is* -- a display
        reporting a clicked pixel must not have to know what a galvo is.

        Exact rather than approximate, for two reasons worth stating because
        both look like off-by-ones otherwise. The displayed frame is already
        cropped of overscan by ``extract_kept_samples``, so displayed column
        ``x`` is total column ``extra_left + x`` and the ``extra_left`` term
        cancels out of the arithmetic. And ``np.repeat`` holds each pixel at one
        voltage for its whole dwell, so there is no half-pixel centre to add:
        this is the voltage that pixel was measured at.

        The caller is responsible for reading this off the ``ScanGroup`` in a
        dataset's provenance rather than the live one. Blocks are shared and
        mutable, so a scan amplitude changed since the image was taken would
        otherwise send the galvos somewhere the picture never looked.
        """
        fast_amp = max(float(self.fast_axis_amplitude), 1e-6)
        slow_amp = max(float(self.slow_axis_amplitude), 1e-6)
        fast_step = (2.0 * fast_amp) / float(self.x_pixels)
        fast_v = float(self.fast_axis_offset) - fast_amp + float(x) * fast_step
        slow_v = (
            float(self.slow_axis_offset)
            + (-1.0 + 2.0 * float(y) / float(self.y_pixels)) * slow_amp
        )
        return fast_v, slow_v


@block
@dataclass
class DaqGroup(Group):
    label: ClassVar[str] = "DAQ"

    sample_rate_hz: float = float_field(
        "Sample Rate (Hz)",
        1_000_000.0,
        minimum=1.0,
        maximum=5_000_000.0,
        step=1_000.0,
        tooltip="DAQ AO sample rate in Hz; the FLIM pixel clock divides down from it",
    )


@block
@dataclass
class ModulationGroup(Group):
    label: ClassVar[str] = "Modulation"

    masks: tuple[Mask, ...] = masks_field(
        "Masks",
        tooltip="Masks driving digital output lines during the scan. "
        "Draw one in the Mask Editor to add it here",
    )


# --------------------------------------------------------------------------- #
# Single-modality blocks                                                       #
# --------------------------------------------------------------------------- #


@block
@dataclass
class PointGroup(Group):
    """Where a point acquisition happens.

    Its own block rather than a field on the spectrum block, because what the
    galvos do and what the detector does are configured independently -- and a
    future point program that is not spectroscopy declares this one and not the
    other.
    """

    label: ClassVar[str] = "Point"

    target: Point = point_field(
        "Target",
        tooltip="Galvo position to park at. Pick it off an image, or type volts directly",
    )


@block
@dataclass
class SpectrumGroup(Group):
    """A stand-in spectrometer, until there is a real one to configure.

    Deterministic in the same way ``SignalGroup`` is: the spectrum is a function
    of (seed, point, frame index), so the same spot gives the same trace on
    every run and two different spots visibly differ. ``num_frames`` is here
    rather than in a block of its own for the same reason it is in
    ``ScanGroup`` -- it decides how the acquisition is done.
    """

    label: ClassVar[str] = "Spectrum"

    num_frames: int = int_field(
        "Frames", 1, minimum=1, tooltip="Number of spectra to capture"
    )
    integration_ms: int = int_field(
        "Integration (ms)",
        500,
        minimum=0,
        tooltip="Dwell per spectrum, standing in for detector integration time",
    )
    n_points: int = int_field(
        "Points", 1024, minimum=16, maximum=65536, tooltip="Samples along the spectral axis"
    )
    n_peaks: int = int_field(
        "Peaks", 5, minimum=0, maximum=64, tooltip="How many bands to synthesise"
    )
    noise_level: float = float_field(
        "Noise Level", 0.03, minimum=0.0, step=0.01, tooltip="Gaussian noise added per point"
    )
    seed: int = int_field(
        "Seed", 1234, minimum=0, tooltip="Same seed and point give the same spectrum"
    )


@block
@dataclass
class SplitGroup(Group):
    label: ClassVar[str] = "Split"

    t0_samples: int = int_field(
        "t0 Samples", 1, minimum=1, tooltip="Number of samples in the first subpixel window"
    )
    t1_samples: int = int_field(
        "t1 Samples", 0, minimum=0, tooltip="Number of samples to discard between t0 and t2"
    )


@block
@dataclass
class TriggerGroup(Group):
    label: ClassVar[str] = "Triggers"

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


@block
@dataclass
class HistogramGroup(Group):
    label: ClassVar[str] = "Histogram"

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
# Simulation                                                                   #
# --------------------------------------------------------------------------- #

#: Pattern names offered by the simulated program, in menu order.
PATTERNS = ("cells", "rings", "gradient", "checkerboard", "flat")


@block
@dataclass
class FrameGroup(Group):
    """The shape of a simulated frame -- what ScanGroup decides on a real rig."""

    label: ClassVar[str] = "Frame"

    num_frames: int = int_field(
        "Frames", 1, minimum=1, tooltip="Number of frames to capture"
    )
    x_pixels: int = int_field("X Pixels", 256, minimum=8, tooltip="Frame width in pixels")
    y_pixels: int = int_field("Y Pixels", 256, minimum=8, tooltip="Frame height in pixels")
    channels: int = int_field(
        "Channels", 2, minimum=1, maximum=16, tooltip="How many detector channels to fake"
    )


@block
@dataclass
class SignalGroup(Group):
    """What the fake detector sees."""

    label: ClassVar[str] = "Signal"

    pattern: str = choice_field(
        "Pattern",
        "cells",
        choices=PATTERNS,
        tooltip="cells drift like a sample, the rest are test targets",
    )
    signal_level: float = float_field(
        "Signal Level", 1.0, minimum=0.0, tooltip="Peak brightness before noise"
    )
    noise_level: float = float_field(
        "Noise Level", 0.03, minimum=0.0, step=0.01, tooltip="Gaussian noise added per pixel"
    )
    drift_pixels_per_frame: float = float_field(
        "Drift (px/frame)", 1.5, minimum=0.0, tooltip="How far the pattern moves each frame"
    )
    mask_gain: float = float_field(
        "Mask Gain",
        0.5,
        minimum=0.0,
        tooltip="Extra brightness inside bound masks, standing in for stimulation",
    )
    seed: int = int_field(
        "Seed", 1234, minimum=0, tooltip="Same seed and frame index give the same pixels"
    )


@block
@dataclass
class PacingGroup(Group):
    """Simulation's stand-in for how long an acquisition takes."""

    label: ClassVar[str] = "Pacing"

    frame_interval_ms: int = int_field(
        "Frame Interval (ms)",
        100,
        minimum=0,
        tooltip="Pause between frames, standing in for acquisition time",
    )
