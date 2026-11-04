"""How program input keys were spelled in older sessions, and the keys that replaced them."""

from __future__ import annotations

import re

# Each program input key that named its section, as ``<section>.<field>``, and the key that
# replaced it. Keys that named the same value in two sections now share one.
SECTIONED_INPUT_KEYS = {
    "camera.exposure_ms": "camera_exposure_ms",
    "camera.gain_pct": "camera_gain_pct",
    "camera.binned": "camera_binned",
    "ccd_readout.exposure_s": "ccd_exposure_s",
    "ccd_readout.frames_to_average": "ccd_frames_to_average",
    "daq.sample_rate_hz": "sample_rate_hz",
    "focus_search.search_range_um": "focus_search_range_um",
    "focus_search.initial_step_um": "focus_initial_step_um",
    "focus_search.final_step_um": "focus_final_step_um",
    "frame_count.num_frames": "num_frames",
    "histogram.laser_frequency_mhz": "laser_frequency_mhz",
    "histogram.histogram_bins": "histogram_bins",
    "histogram.histogram_binwidth_ps": "histogram_binwidth_ps",
    "histogram.frame_settle_s": "frame_settle_s",
    "modulation.lines": "mask_lines",
    "mosaic.x_tiles": "x_tiles",
    "mosaic.y_tiles": "y_tiles",
    "mosaic.spacing_um": "tile_spacing_um",
    "point.target": "galvo_target",
    "roi.center_fraction": "center_region_fraction",
    "scan.x_pixels": "x_pixels",
    "scan.y_pixels": "y_pixels",
    "scan.extra_left": "extra_left",
    "scan.extra_right": "extra_right",
    "scan.fast_axis_offset": "fast_axis_offset",
    "scan.fast_axis_amplitude": "fast_axis_amplitude",
    "scan.slow_axis_offset": "slow_axis_offset",
    "scan.slow_axis_amplitude": "slow_axis_amplitude",
    "scan.dwell_time_us": "dwell_time_us",
    "split.t0_samples": "t0_samples",
    "split.t1_samples": "t1_samples",
    "trigger.frame_trigger_pfi": "frame_trigger_pfi",
    "trigger.pixel_clock_ctr": "pixel_clock_ctr",
    "trigger.pixel_clock_pfi": "pixel_clock_pfi",
    "frame.x_pixels": "x_pixels",
    "frame.y_pixels": "y_pixels",
    "frame.channels": "simulated_channels",
    "pacing.frame_interval_ms": "frame_interval_ms",
    "signal.pattern": "signal_pattern",
    "signal.signal_level": "signal_level",
    "signal.noise_level": "noise_level",
    "signal.drift_pixels_per_frame": "drift_pixels_per_frame",
    "signal.mask_gain": "mask_gain",
    "specimen.um_per_pixel": "um_per_pixel",
    "specimen.noise_level": "noise_level",
}


def legacy_input_key(path: str) -> str:
    """The sectioned program input key that replaced ``<Block>.<field>``, as in
    ``"ScanGroup.dwell_time_us"`` to ``"scan.dwell_time_us"``."""
    block, _, field = path.partition(".")
    prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", block.removesuffix("Group")).lower()
    return f"{prefix}.{field}"


def unsectioned_input_key(key: str) -> str:
    """The program input key that replaced the sectioned ``key``, or ``key`` if none did."""
    return SECTIONED_INPUT_KEYS.get(key, key)
