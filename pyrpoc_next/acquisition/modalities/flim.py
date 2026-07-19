"""FLIM: a galvo raster synchronized to a TimeTagger that histograms photon delays.

Requires the NI-DAQ (galvo scan + pixel-clock/frame-trigger markers) and the
TimeTagger (per-pixel decay histograms). Real hardware only. FLIM does not realize
the mask modifier, and its manifest says so honestly.
"""

from __future__ import annotations

import time

import numpy as np

from pyrpoc_next.acquisition.modalities.base import Modality, modality_registry
from pyrpoc_next.acquisition.modalities.scanner import acquisition_parameters, scan_parameters
from pyrpoc_next.instruments.scanning import RasterScan
from pyrpoc_next.structs.keys import InstrumentKey, ModalityKey
from pyrpoc_next.structs.manifest import ModalityManifest
from pyrpoc_next.structs.parameters import NumberParameter
from pyrpoc_next.structs.parcels import HistogramCubeParcel, ImageFrameParcel

frame_settle_s = 5e-3


def flim_daq_parameters():
    return [
        NumberParameter(label="Sample Rate (Hz)", default=1_000_000.0, minimum=1, maximum=5_000_000, step=1000),
        NumberParameter(label="Fast Axis AO", default=0, minimum=0, maximum=31, number_type=int),
        NumberParameter(label="Slow Axis AO", default=1, minimum=0, maximum=31, number_type=int),
        NumberParameter(label="Frame Trigger PFI", default=0, minimum=0, number_type=int),
        NumberParameter(label="Pixel Clock Counter", default=0, minimum=0, number_type=int),
        NumberParameter(label="Pixel Clock PFI", default=1, minimum=0, number_type=int),
    ]


def timetagger_parameters():
    return [
        NumberParameter(label="Laser Channel", default=1, number_type=int),
        NumberParameter(label="Detector Channel", default=2, number_type=int),
        NumberParameter(label="Pixel Channel", default=3, number_type=int),
        NumberParameter(label="Frame Channel", default=4, number_type=int),
        NumberParameter(label="Laser Frequency (MHz)", default=80.0, minimum=1e-6),
        NumberParameter(label="Histogram Bins", default=125, minimum=1, number_type=int),
        NumberParameter(label="Histogram Bin Width (ps)", default=100, minimum=1, number_type=int),
        NumberParameter(label="Laser Trigger (V)", default=0.05),
        NumberParameter(label="Detector Trigger (V)", default=0.2),
        NumberParameter(label="Pixel Trigger (V)", default=0.5),
        NumberParameter(label="Frame Trigger (V)", default=0.5),
        NumberParameter(label="Laser Input Delay (ps)", default=0, number_type=int),
    ]


def flim_parameter_groups():
    return {
        "scan": scan_parameters(),
        "daq": flim_daq_parameters(),
        "timetagger": timetagger_parameters(),
        "acquisition": acquisition_parameters(),
    }


def read_histogram_cube(measurement, scan: RasterScan, n_bins: int) -> np.ndarray:
    """Read the just-scanned Flim frame into a clipped (H, W, bins) cube."""
    frame = measurement.getCurrentFrameEx()
    flat = np.asarray(frame.getHistograms(), dtype=np.float32)
    cube = flat.reshape(scan.y_pixels, scan.total_x, n_bins)
    return cube[:, scan.extra_left : scan.extra_left + scan.x_pixels, :]


@modality_registry.register
class FlimModality(Modality):
    """Fluorescence-lifetime imaging."""

    key = ModalityKey.flim
    manifest = ModalityManifest(
        key=ModalityKey.flim,
        display_name="FLIM",
        emitted_parcels=(ImageFrameParcel, HistogramCubeParcel),
        required_instruments=(InstrumentKey.ni_daq, InstrumentKey.time_tagger),
        realizable_modifiers=(),
        parameter_groups=flim_parameter_groups(),
    )

    def geometry(self, values: dict) -> RasterScan:
        """Build the scan geometry; FLIM reads no analog input, so ai_channels is empty."""
        return RasterScan(
            x_pixels=values["X Pixels"], y_pixels=values["Y Pixels"],
            extra_left=values["Extra Steps Left"], extra_right=values["Extra Steps Right"],
            fast_axis_offset=values["Fast Axis Offset"], fast_axis_amplitude=values["Fast Axis Amplitude"],
            slow_axis_offset=values["Slow Axis Offset"], slow_axis_amplitude=values["Slow Axis Amplitude"],
            dwell_time_us=values["Dwell Time (us)"], sample_rate_hz=values["Sample Rate (Hz)"],
            fast_axis_ao=values["Fast Axis AO"], slow_axis_ao=values["Slow Axis AO"], ai_channels=[],
        )

    def acquire_frame(self, index: int) -> list:
        values = self.values
        daq = self.instruments[InstrumentKey.ni_daq]
        tagger = self.instruments[InstrumentKey.time_tagger]
        scan = self.geometry(values)
        n_bins = values["Histogram Bins"]
        bin_width = values["Histogram Bin Width (ps)"]
        laser_period = round(1e6 / values["Laser Frequency (MHz)"])
        n_pixels = scan.total_x * scan.y_pixels

        tagger.create_tagger()
        tagger.configure_for_flim(
            values["Laser Channel"], values["Detector Channel"], values["Pixel Channel"], values["Frame Channel"],
            values["Laser Trigger (V)"], values["Detector Trigger (V)"], values["Pixel Trigger (V)"],
            values["Frame Trigger (V)"], laser_input_delay_ps=values["Laser Input Delay (ps)"],
        )
        measurement = tagger.create_flim_measurement(
            values["Laser Channel"], values["Detector Channel"], values["Pixel Channel"],
            values["Frame Channel"], n_pixels, n_bins, bin_width,
        )
        try:
            daq.flim_scan(scan, values["Frame Trigger PFI"], values["Pixel Clock Counter"], values["Pixel Clock PFI"])
            time.sleep(frame_settle_s)
            cube = read_histogram_cube(measurement, scan, n_bins)
        finally:
            tagger.free_tagger()

        intensity = cube.sum(axis=2)[np.newaxis].astype(np.float32, copy=False)
        return [
            ImageFrameParcel(data=intensity, channel_labels=["intensity"]),
            HistogramCubeParcel(data=cube, bin_width_ps=float(bin_width), laser_period_ps=float(laser_period)),
        ]
