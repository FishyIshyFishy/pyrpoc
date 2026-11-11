"""``required``: an earlier output a run reads, which preflight has made sure is set."""

from __future__ import annotations

from typing import TypeVar

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_input import ProgramInput

T = TypeVar("T")


def required(ctx: RunContext, program_input: ProgramInput[T | None]) -> T:
    """The run's value of ``program_input``, which is None only if preflight let a run start
    without it."""
    value = ctx.inputs[program_input]
    if value is None:
        raise RuntimeError(f"{program_input.key!r} is not set, yet the run started")
    return value
