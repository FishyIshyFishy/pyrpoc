"""The CCD readout a run chooses: exposure and averaging."""

from __future__ import annotations

from pyrpoc.utils.value_rules import float_input, int_input

CCD_EXPOSURE_S = float_input(
    "ccd_exposure_s",
    "Exposure (s)",
    0.1,
    minimum=0.0,
    step=0.01,
    decimals=4,
    tooltip="Per frame",
)
CCD_FRAMES_TO_AVERAGE = int_input(
    "ccd_frames_to_average",
    "Frames to Average",
    1,
    minimum=1,
    tooltip="Reads accumulated on the camera and averaged into one spectrum",
)
