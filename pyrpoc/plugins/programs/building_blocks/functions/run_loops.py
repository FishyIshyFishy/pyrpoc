"""Loops a run steps through: reporting each step to the status line and
stopping when the run is stopped."""

from __future__ import annotations

from collections.abc import Iterator

from pyrpoc.core.program import RunContext


def count_off(ctx: RunContext, count: int, noun: str) -> Iterator[int]:
    """``range(count)``, reporting each step and stopping when the run is stopped."""
    for index in range(count):
        ctx.check_cancel()
        ctx.status(f"{noun} {index + 1}/{count}")
        yield index
