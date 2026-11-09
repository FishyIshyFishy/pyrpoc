"""Multisite: measure the scanner against the stage, map a dish, then run routines on chosen
regions of it over time. Programs import only what this module exports."""

from __future__ import annotations

from .bindings.monitor import MONITOR_INPUTS, MONITOR_LIVE, TREATMENT_INPUTS, monitor_routine
from .bindings.programs import CalibrationProgram, LoopProgram, SurveyProgram
from .bindings.rig import RigMicroscope
from .interfaces.microscope import DigitalLine, Microscope
from .stages.loop.routine import RoutineSpec

__all__ = [
    "MONITOR_INPUTS",
    "MONITOR_LIVE",
    "TREATMENT_INPUTS",
    "CalibrationProgram",
    "DigitalLine",
    "LoopProgram",
    "Microscope",
    "RigMicroscope",
    "RoutineSpec",
    "SurveyProgram",
    "monitor_routine",
]
