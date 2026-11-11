"""What a multisite step needs before it starts, beyond what the engine judges for every program:
a folder holding only the steps it reads, and for a loop, units it can start on."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pyrpoc.core.program import PreflightContext
from pyrpoc.core.program_input import ProgramInput
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.plugins.program_io.inputs import SURVEY, UNITS

from ..stages.loop.orchestrator import fits
from .scratch import Scratch


def clean_folder(
    ctx: PreflightContext, reads: Sequence[ProgramInput[Any]], carries_on: bool
) -> list[str]:
    """Refuses a folder holding any recording the step does not read, so each read comes from
    the only recording that can supply it. With ``carries_on``, the recordings of a loop cut
    short, which its scratch names, may stay, so a new run can carry it on."""
    read: list[ProgramOutput | None] = [ctx.inputs[each] for each in reads]
    allowed = {each.recording.name for each in read if each is not None}
    if carries_on and ctx.folder is not None and Scratch(ctx.folder).exists():
        allowed |= set(Scratch(ctx.folder).recordings())
    # Failed and unreadable recordings count too: their files are in the folder all the same.
    extra = [name for name in ctx.recordings_in_folder() if name not in allowed]
    if not extra:
        return []
    return [f"a folder holding only what it reads (move out {', '.join(extra)})"]


def starting_units(ctx: PreflightContext) -> list[str]:
    """Refuses a fresh loop with no units drawn, or with units larger than one field of the
    survey read. A loop carried on starts from its scratch instead, so it is refused if units
    are drawn: they would not be what it starts on."""
    units = ctx.inputs[UNITS]
    if ctx.folder is not None and Scratch(ctx.folder).exists():
        if not units:
            return []
        return [
            "no units drawn, to carry on the loop stopped in this folder "
            "(or choose a new folder to start afresh)"
        ]
    if not units:
        return ["at least one unit drawn over the survey"]
    survey = ctx.inputs[SURVEY]
    # The engine names a survey not yet read among what the run needs.
    if survey is None:
        return []
    too_large = sum(not fits(unit, survey.um_per_px, survey.field_pixels) for unit in units)
    if not too_large:
        return []
    return [f"units that each fit in one field ({too_large} of {len(units)} are larger)"]
