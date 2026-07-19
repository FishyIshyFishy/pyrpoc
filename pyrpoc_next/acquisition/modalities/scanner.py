"""ScannerModality: DRY helpers shared by galvo-raster modalities.

A convenience base, not a framework — it supplies the common scan/daq parameter
groups, builds a RasterScan from coerced values, and turns enabled mask modifiers
into TTL. Modalities inherit it only to avoid copy-paste.
"""

from __future__ import annotations

from pyrpoc_next.acquisition.modalities.base import Modality
from pyrpoc_next.acquisition.modifiers.mask import MaskModifier
from pyrpoc_next.instruments.ni_daq import mask_to_ttl
from pyrpoc_next.instruments.scanning import RasterScan
from pyrpoc_next.structs.keys import InstrumentKey
from pyrpoc_next.structs.parameters import (
    ChannelSelectionParameter,
    CheckboxParameter,
    NumberParameter,
    PathParameter,
)


def scan_parameters():
    return [
        NumberParameter(label="X Pixels", default=256, minimum=8, number_type=int),
        NumberParameter(label="Y Pixels", default=256, minimum=8, number_type=int),
        NumberParameter(label="Extra Steps Left", default=50, minimum=0, number_type=int),
        NumberParameter(label="Extra Steps Right", default=20, minimum=0, number_type=int),
        NumberParameter(label="Fast Axis Offset", default=0.0),
        NumberParameter(label="Fast Axis Amplitude", default=1.0, minimum=1e-6),
        NumberParameter(label="Slow Axis Offset", default=0.0),
        NumberParameter(label="Slow Axis Amplitude", default=1.0, minimum=1e-6),
        NumberParameter(label="Dwell Time (us)", default=2.0, minimum=0.1),
    ]


def daq_parameters():
    return [
        NumberParameter(label="Sample Rate (Hz)", default=100000.0, minimum=1, maximum=5_000_000, step=1000),
        NumberParameter(label="Fast Axis AO", default=0, minimum=0, maximum=31, number_type=int),
        NumberParameter(label="Slow Axis AO", default=1, minimum=0, maximum=31, number_type=int),
        ChannelSelectionParameter(label="Active AI Channels", default=[0], channel_count=9),
    ]


def frames_parameter():
    return NumberParameter(label="Frames", default=1, minimum=1, number_type=int)


def acquisition_parameters():
    return [
        CheckboxParameter(label="Save", default=False),
        PathParameter(label="Save Path", default="acquisition", required=False),
        frames_parameter(),
    ]


def scanner_parameter_groups():
    return {"scan": scan_parameters(), "daq": daq_parameters(), "acquisition": acquisition_parameters()}


class ScannerModality(Modality):
    """Base for modalities that drive a galvo raster."""

    def daq(self):
        return self.instruments[InstrumentKey.ni_daq]

    def build_scan(self, values: dict) -> RasterScan:
        """Build a RasterScan from coerced parameter values."""
        return RasterScan(
            x_pixels=values["X Pixels"],
            y_pixels=values["Y Pixels"],
            extra_left=values["Extra Steps Left"],
            extra_right=values["Extra Steps Right"],
            fast_axis_offset=values["Fast Axis Offset"],
            fast_axis_amplitude=values["Fast Axis Amplitude"],
            slow_axis_offset=values["Slow Axis Offset"],
            slow_axis_amplitude=values["Slow Axis Amplitude"],
            dwell_time_us=values["Dwell Time (us)"],
            sample_rate_hz=values["Sample Rate (Hz)"],
            fast_axis_ao=values["Fast Axis AO"],
            slow_axis_ao=values["Slow Axis AO"],
            ai_channels=values["Active AI Channels"],
        )

    def mask_ttl(self, scan: RasterScan, active_samples: int) -> dict:
        """Turn enabled mask modifiers into DAQ TTL, gated to active_samples per pixel."""
        signals: dict = {}
        device = self.daq().device_name
        for modifier in self.modifiers:
            if isinstance(modifier, MaskModifier) and modifier.mask is not None:
                signals.update(
                    mask_to_ttl(modifier.mask, device, modifier.daq_port, modifier.daq_line, scan,
                                active_samples=active_samples)
                )
        return signals
