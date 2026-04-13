from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.backend_utils.parameter_utils import (
    ChannelSelectionParameter,
    CheckboxParameter,
    NumberParameter,
    PathParameter,
    TextParameter,
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
            tooltip="Extra scan steps at left edge",
            number_type=int,
        ),
        NumberParameter(
            label="Extra Steps Right",
            default=20,
            minimum=0,
            tooltip="Extra scan steps at right edge",
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
    "daq": [
        TextParameter(
            label="DAQ Device",
            default="Dev1",
            tooltip="NI-DAQ device name (e.g. Dev1)",
        ),
        NumberParameter(
            label="Sample Rate (Hz)",
            default=100_000.0,
            minimum=1.0,
            maximum=5_000_000.0,
            step=1_000.0,
            tooltip="DAQ sample rate in Hz",
            number_type=float,
        ),
        NumberParameter(
            label="Fast Axis AO",
            default=0,
            minimum=0,
            maximum=31,
            tooltip="Analog output channel for the fast (X) galvo",
            number_type=int,
        ),
        NumberParameter(
            label="Slow Axis AO",
            default=1,
            minimum=0,
            maximum=31,
            tooltip="Analog output channel for the slow (Y) galvo",
            number_type=int,
        ),
        ChannelSelectionParameter(
            label="Active AI Channels",
            num_channels=9,
            tooltip="Toggle which analog input channels are active",
        ),
    ],
    "timetagger": [
        NumberParameter(
            label="Laser Channel",
            default=1,
            minimum=1,
            tooltip="TimeTagger input channel for laser sync",
            number_type=int,
        ),
        NumberParameter(
            label="Detector Channel",
            default=2,
            minimum=1,
            tooltip="TimeTagger input channel for SPAD detector",
            number_type=int,
        ),
        NumberParameter(
            label="DAQ Trigger Channel",
            default=3,
            minimum=1,
            tooltip="TimeTagger input channel for the DAQ frame-start trigger",
            number_type=int,
        ),
        NumberParameter(
            label="DAQ Trigger PFI Line",
            default=0,
            minimum=0,
            tooltip="NI-DAQ PFI line number exported as the frame-start trigger",
            number_type=int,
        ),
        NumberParameter(
            label="Laser Frequency MHz",
            default=80.0,
            minimum=0.001,
            tooltip="Laser repetition rate in MHz (used to fold delays)",
            number_type=float,
        ),
        NumberParameter(
            label="Laser Trigger V",
            default=0.05,
            tooltip="Trigger threshold for laser sync channel (V)",
            number_type=float,
        ),
        NumberParameter(
            label="Detector Trigger V",
            default=0.2,
            tooltip="Trigger threshold for SPAD detector channel (V)",
            number_type=float,
        ),
        NumberParameter(
            label="Trigger V",
            default=0.2,
            tooltip="Trigger threshold for DAQ trigger channel (V)",
            number_type=float,
        ),
        NumberParameter(
            label="Laser Event Divider",
            default=1,
            minimum=1,
            tooltip="Only keep 1 in N laser sync events (reduces data rate)",
            number_type=int,
        ),
    ],
    "acquisition": [
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
            tooltip="Base name/path for saved files",
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
class FlimParameters(AcquisitionParameters):
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
    # daq
    device_name: str
    sample_rate_hz: float
    fast_axis_ao: int
    slow_axis_ao: int
    active_ai_channels: tuple[int, ...]
    # timetagger
    laser_channel: int
    detector_channel: int
    daq_trigger_channel: int
    daq_trigger_pfi_line: int
    laser_frequency_mhz: float
    laser_trigger_v: float
    detector_trigger_v: float
    trigger_v: float
    laser_event_divider: int
    # acquisition
    save_enabled: bool
    save_path: str
    num_frames: int

    @classmethod
    def from_dict(cls, p: dict) -> FlimParameters:
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
            device_name=str(p.get("DAQ Device", "Dev1")) or "Dev1",
            sample_rate_hz=float(p.get("Sample Rate (Hz)", 100_000.0)),
            fast_axis_ao=int(p.get("Fast Axis AO", 0)),
            slow_axis_ao=int(p.get("Slow Axis AO", 1)),
            active_ai_channels=tuple(int(c) for c in p.get("Active AI Channels", list(range(9)))),
            laser_channel=int(p["Laser Channel"]),
            detector_channel=int(p["Detector Channel"]),
            daq_trigger_channel=int(p["DAQ Trigger Channel"]),
            daq_trigger_pfi_line=int(p["DAQ Trigger PFI Line"]),
            laser_frequency_mhz=float(p["Laser Frequency MHz"]),
            laser_trigger_v=float(p["Laser Trigger V"]),
            detector_trigger_v=float(p["Detector Trigger V"]),
            trigger_v=float(p["Trigger V"]),
            laser_event_divider=int(p["Laser Event Divider"]),
            save_enabled=bool(p.get("save_enabled", False)),
            save_path=str(p.get("save_path", "acquisition")),
            num_frames=int(p.get("num_frames", 1)),
        )
