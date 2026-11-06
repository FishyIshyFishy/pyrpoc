"""``CalibrationMeasured`` and ``SurveyFinished``: a pipeline step starts only on earlier steps
that ran to their end."""

# A step stopped partway still ends without an error, so it is offered as a read; these checks
# refuse it before the run opens, rather than the run failing on it.

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.check import Check, CheckContext
from pyrpoc.plugins.program_io.outputs import GalvoCalibrationOutput, SurveyOutput


@dataclass(frozen=True)
class CalibrationMeasured(Check):
    """Refuses a calibration read whose run was stopped before it measured."""

    def blockers(self, ctx: CheckContext) -> list[str]:
        if ctx.earlier("calibration", GalvoCalibrationOutput).calibration is not None:
            return []
        name = ctx.reading()["calibration"]
        return [f"a calibration that finished ({name} was stopped before it measured)"]


@dataclass(frozen=True)
class SurveyFinished(Check):
    """Refuses a survey read whose run was stopped before it made its overview."""

    def blockers(self, ctx: CheckContext) -> list[str]:
        if ctx.earlier("survey", SurveyOutput).finding is not None:
            return []
        name = ctx.reading()["survey"]
        return [f"a survey that finished ({name} was stopped before its overview)"]
