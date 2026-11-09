"""What a survey covers, and how each of its tiles is scanned."""

from __future__ import annotations

from pyrpoc.utils.value_rules import float_input, int_input

SURVEY_WIDTH_UM = float_input(
    "survey_width_um",
    "Width (µm)",
    1000.0,
    minimum=1.0,
    step=100.0,
    decimals=1,
    tooltip="How far the survey reaches in x from where the stage is now, its top-left corner",
)
SURVEY_HEIGHT_UM = float_input(
    "survey_height_um",
    "Height (µm)",
    1000.0,
    minimum=1.0,
    step=100.0,
    decimals=1,
    tooltip="How far the survey reaches in y from where the stage is now, its top-left corner",
)
FIELD_UM = float_input(
    "field_um",
    "Field (µm)",
    150.0,
    minimum=1.0,
    step=10.0,
    decimals=1,
    tooltip="The width and height of one tile on the sample; the calibration turns it into volts",
)
FIELD_PIXELS = int_input(
    "field_pixels",
    "Pixels per Field",
    256,
    minimum=16,
    tooltip="How many pixels across each tile is scanned at",
)
TILE_OVERLAP = float_input(
    "tile_overlap",
    "Tile Overlap",
    0.15,
    minimum=0.0,
    maximum=0.9,
    step=0.05,
    decimals=2,
    tooltip="The fraction of a tile its neighbours share, which stitching matches on",
)
