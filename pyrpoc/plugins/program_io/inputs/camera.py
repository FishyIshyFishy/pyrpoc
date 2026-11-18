"""The program inputs of the camera: exposure, gain and binning."""

from __future__ import annotations

from pyrpoc.utils.value_rules import bool_input, float_input, int_input

# Program inputs rather than device configuration: they change with the
# sample, not with how the camera is wired.
CAMERA_EXPOSURE_MS = float_input(
    "camera_exposure_ms",
    "Exposure (ms)",
    10.0,
    minimum=0.001,
    decimals=3,
    tooltip="Exposure time per frame",
)
CAMERA_GAIN_PCT = int_input(
    "camera_gain_pct",
    "Gain (%)",
    100,
    minimum=100,
    tooltip="Analog gain; 100 is unity, the camera sets the top",
)
CAMERA_BINNED = bool_input(
    "camera_binned",
    "Bin 2x2",
    False,
    tooltip="Half the width and height, for faster and brighter frames",
)
