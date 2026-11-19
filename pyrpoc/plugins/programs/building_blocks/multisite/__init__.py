"""Multisite: measure the scanner against the stage, map a dish, then run routines on chosen
regions of it over time; or, with the same calibration, image a mosaic around the stage.
Programs import only what this module exports."""

from __future__ import annotations

from .bindings.monitor import MONITOR_INPUTS, TREATMENT_INPUTS, monitor_routine
from .bindings.programs import CalibrationProgram, LoopProgram, MosaicProgram, SurveyProgram
from .bindings.rig import RigMicroscope
from .interfaces.microscope import DigitalLine, Microscope
from .stages.loop.routine import RoutineSpec

__all__ = [
    "MONITOR_INPUTS",
    "TREATMENT_INPUTS",
    "CalibrationProgram",
    "DigitalLine",
    "LoopProgram",
    "Microscope",
    "MosaicProgram",
    "RigMicroscope",
    "RoutineSpec",
    "SurveyProgram",
    "monitor_routine",
]
