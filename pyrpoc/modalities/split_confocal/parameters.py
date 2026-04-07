from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.backend_utils.parameter_utils import (
    CheckboxParameter,
    NumberParameter,
    PathParameter,
)

from ..base_modality import AcquisitionParameters

PARAMETERS = {
    "scan": [
        NumberParameter(
            label="X Pixels",
            default=512,
            minimum=8,
            tooltip="Number of pixels in X",
            number_type=int,
        ),
        NumberParameter(
            label="Y Pixels",
            default=512,
            minimum=8,
            tooltip="Number of pixels in Y",
            number_type=int,
        ),
        NumberParameter(
            label="Extra Steps Left",
            default=300,
            minimum=0,
            tooltip="Extra scan steps at left edge (stored only for now)",
            number_type=int,
        ),
        NumberParameter(
            label="Extra Steps Right",
            default=20,
            minimum=0,
            tooltip="Extra scan steps at right edge (stored only for now)",
            number_type=int,
        ),
        NumberParameter(
            label="Fast Axis Offset",
            default=0.0,
            tooltip="Fast-axis offset",
            number_type=float,
        ),
        NumberParameter(
            label="Fast Axis Amplitude",
            default=1.0,
            minimum=1e-6,
            tooltip="Fast-axis amplitude",
            number_type=float,
        ),
        NumberParameter(
            label="Slow Axis Offset",
            default=0.0,
            tooltip="Slow-axis offset",
            number_type=float,
        ),
        NumberParameter(
            label="Slow Axis Amplitude",
            default=1.0,
            minimum=1e-6,
            tooltip="Slow-axis amplitude",
            number_type=float,
        ),
        NumberParameter(
            label="Dwell Time (us)",
            default=2.0,
            minimum=0.1,
            tooltip="Pixel dwell time",
            number_type=float,
        ),
    ],
    "acquisition": [
        NumberParameter(
            label="t0 Samples",
            default=1,
            minimum=1,
            tooltip="Number of samples in the first subpixel window",
            number_type=int,
        ),
        NumberParameter(
            label="t1 Samples",
            default=0,
            minimum=0,
            tooltip="Number of samples to discard between t0 and t2",
            number_type=int,
        ),
        CheckboxParameter(
            label="save_enabled",
            display_label="save_enabled",
            default=False,
            required=False,
            tooltip="Enable saving frames and acquisition metadata",
        ),
        PathParameter(
            label="save_path",
            display_label="save_path",
            default="acquisition",
            required=False,
            tooltip="Base name/path for saved TIFF files (e.g. /dir/acquisition)",
        ),
        NumberParameter(
            label="num_frames",
            display_label="num_frames",
            default=1,
            required=False,
            minimum=1,
            tooltip="Number of frames to capture",
            number_type=int,
        ),
    ],
}


@dataclass(frozen=True)
class SplitConfocalParameters(AcquisitionParameters):
    # scan
    x_pixels: int
    y_pixels: int
    extra_left: int
    extra_right: int
    fast_axis_offset: float
    fast_axis_amplitude: float
    slow_axis_offset: float
    slow_axis_amplitude: float
    dwell_time_us: float
    # split timing
    t0_samples: int
    t1_samples: int
    # acquisition
    save_enabled: bool
    save_path: str
    num_frames: int

    @classmethod
    def from_dict(cls, p: dict) -> SplitConfocalParameters:
        t0 = int(p["t0 Samples"])
        t1 = int(p["t1 Samples"])
        if t0 < 1:
            raise ValueError("t0 Samples must be >= 1")
        if t1 < 0:
            raise ValueError("t1 Samples must be >= 0")
        return cls(
            x_pixels=int(p["X Pixels"]),
            y_pixels=int(p["Y Pixels"]),
            extra_left=int(p["Extra Steps Left"]),
            extra_right=int(p["Extra Steps Right"]),
            fast_axis_offset=float(p["Fast Axis Offset"]),
            fast_axis_amplitude=max(float(p["Fast Axis Amplitude"]), 1e-6),
            slow_axis_offset=float(p["Slow Axis Offset"]),
            slow_axis_amplitude=max(float(p["Slow Axis Amplitude"]), 1e-6),
            dwell_time_us=float(p["Dwell Time (us)"]),
            t0_samples=t0,
            t1_samples=t1,
            save_enabled=bool(p.get("save_enabled", False)),
            save_path=str(p.get("save_path", "acquisition")),
            num_frames=int(p.get("num_frames", 1)),
        )
