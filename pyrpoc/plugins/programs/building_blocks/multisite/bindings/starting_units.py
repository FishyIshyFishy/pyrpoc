"""``StartingUnits``: a fresh loop starts only on units drawn over the survey, each fitting in
one of its fields."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.check import Check, CheckContext
from pyrpoc.plugins.program_io.inputs import UNITS
from pyrpoc.plugins.program_io.outputs import SurveyOutput

from ..stages.loop.orchestrator import fits
from .scratch import Scratch


@dataclass(frozen=True)
class StartingUnits(Check):
    """Refuses a loop with no units drawn, or with units larger than one field of the survey
    read. A loop carried on starts from its scratch, not the units drawn, so it is never
    refused for them."""

    def blockers(self, ctx: CheckContext) -> list[str]:
        if ctx.folder is not None and Scratch(ctx.folder).exists():
            return []
        units = ctx.inputs[UNITS]
        if not units:
            return ["at least one unit drawn over the survey"]
        survey = ctx.earlier("survey", SurveyOutput)
        # An unfinished survey may hold no tile to size a field by; SurveyFinished refuses it.
        if survey.finding is None:
            return []
        pixels = survey[0].shape[-1]
        too_large = sum(not fits(unit, survey.um_per_px, pixels) for unit in units)
        if not too_large:
            return []
        return [f"units that each fit in one field ({too_large} of {len(units)} are larger)"]
