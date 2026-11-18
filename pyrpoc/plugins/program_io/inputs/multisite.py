"""The program inputs of multisite: what a survey covers, how a loop runs, the units it runs
on, and the earlier calibration and survey it reads."""

from __future__ import annotations

from pyrpoc.core.program_input import (
    ProgramInput,
    SidecarProgramInput,
    declare,
    declare_sidecar,
)
from pyrpoc.plugins.program_io.outputs import GalvoCalibrationOutput, SurveyOutput, Unit
from pyrpoc.utils.value_rules import (
    EarlierOutputField,
    TextField,
    bool_input,
    float_input,
    int_input,
)

from .values.units import UnitsField, UnitsJson

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

PERIOD_S = float_input(
    "loop_period_s",
    "Period (s)",
    60.0,
    minimum=1.0,
    step=10.0,
    decimals=1,
    tooltip="How long after one scan of a unit its next is due",
)
DURATION_MIN = float_input(
    "loop_duration_min",
    "Duration (min)",
    30.0,
    minimum=0.1,
    step=5.0,
    decimals=1,
    tooltip="How long each unit is monitored for",
)
FOCUS_CHECK_S = float_input(
    "loop_focus_check_s",
    "Focus Check (s)",
    300.0,
    minimum=10.0,
    step=60.0,
    decimals=0,
    tooltip=(
        "The longest between focus checks; they come sooner while the drift is not yet known "
        "well enough to predict focus to the final step"
    ),
)
TREAT = bool_input(
    "loop_treat",
    "Treat at Start",
    False,
    tooltip=(
        "Before monitoring, scan each unit once: the treated group's with the line driven, "
        "the rest's without it as a sham, so both groups take the same imaging light"
    ),
)
TREATED_GROUP = declare(
    "loop_treated_group",
    TextField("Treated Group", "The group whose units are treated at the start"),
    "treated",
)
TREATMENT_PORT = int_input(
    "loop_treatment_port", "Treatment Port", 0, minimum=0, tooltip="The DAQ port of the line"
)
TREATMENT_LINE = int_input(
    "loop_treatment_line",
    "Treatment Line",
    0,
    minimum=0,
    tooltip="The digital line driven high over a unit while it is treated",
)

UNITS: SidecarProgramInput[tuple[Unit, ...]] = declare_sidecar(
    "units",
    UnitsField("Units", "The regions of the dish a loop runs on, drawn over the survey"),
    (),
    # Thousands of outlines would be rewritten on every autosave.
    restored=False,
    file=UnitsJson(),
)

# Not restored: the engine sets each from the save folder, the newest there of its type.
CALIBRATION: ProgramInput[GalvoCalibrationOutput | None] = declare(
    "calibration",
    EarlierOutputField(
        "Calibration",
        "The galvo calibration saved in this folder",
        output_type=GalvoCalibrationOutput,
    ),
    None,
    restored=False,
)
SURVEY: ProgramInput[SurveyOutput | None] = declare(
    "survey",
    EarlierOutputField("Survey", "The survey saved in this folder", output_type=SurveyOutput),
    None,
    restored=False,
)
