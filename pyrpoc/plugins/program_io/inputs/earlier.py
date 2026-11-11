"""``CALIBRATION`` and ``SURVEY``: outputs of earlier recordings in the save folder that later
multisite steps read, as program inputs."""

from __future__ import annotations

from pyrpoc.core.program_input import ProgramInput, declare
from pyrpoc.plugins.program_io.outputs import GalvoCalibrationOutput, SurveyOutput
from pyrpoc.utils.value_rules import EarlierOutputField

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
