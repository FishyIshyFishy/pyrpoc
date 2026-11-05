"""Multisite: measure the scanner against the stage, map a dish, then run routines on chosen
regions of it over time. Programs import only what this module exports."""

from __future__ import annotations

from .bindings.calibration import CALIBRATION_EMITS, run_calibration
from .bindings.clean_folder import CleanFolder
from .bindings.loop import LOOP_EMITS, LOOP_FOCUS_INPUTS, LOOP_READS, STEERING, run_loop
from .bindings.monitor import MONITOR_INPUTS, MONITOR_LIVE, TREATMENT_INPUTS, monitor_routine
from .bindings.rig import RigMicroscope
from .bindings.survey import (
    REGION_INPUTS,
    SURVEY_EMITS,
    SURVEY_FOCUS_INPUTS,
    SURVEY_READS,
    TILE_INPUTS,
    run_survey,
)
from .interfaces.microscope import DigitalLine, Microscope

__all__ = [
    "CALIBRATION_EMITS",
    "LOOP_EMITS",
    "LOOP_FOCUS_INPUTS",
    "LOOP_READS",
    "MONITOR_INPUTS",
    "MONITOR_LIVE",
    "REGION_INPUTS",
    "STEERING",
    "SURVEY_EMITS",
    "SURVEY_FOCUS_INPUTS",
    "SURVEY_READS",
    "TILE_INPUTS",
    "TREATMENT_INPUTS",
    "CleanFolder",
    "DigitalLine",
    "Microscope",
    "RigMicroscope",
    "monitor_routine",
    "run_calibration",
    "run_loop",
    "run_survey",
]
