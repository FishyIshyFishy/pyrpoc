"""``CleanFolder``: a pipeline step starts only in a folder holding just the earlier steps it
reads, and for a loop, the runs of a loop it carries on."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.check import Check, CheckContext

from .scratch import Scratch


@dataclass(frozen=True)
class CleanFolder(Check):
    """Refuses a folder holding any recording the program does not read. Each read then comes
    from the only recording that can supply it, and a step with no reads needs an empty
    folder. With ``carries_on``, the recordings of a loop cut short, which its scratch names,
    are allowed too, so a new run can carry it on."""

    carries_on: bool

    def blockers(self, ctx: CheckContext) -> list[str]:
        allowed = set(ctx.reading().values())
        if self.carries_on and ctx.folder is not None and Scratch(ctx.folder).exists():
            allowed |= set(Scratch(ctx.folder).recordings())
        # Failed and unreadable recordings count too: their files are in the folder all the same.
        extra = [name for name in ctx.recordings_in_folder() if name not in allowed]
        if not extra:
            return []
        return [f"a folder holding only what it reads (move out {', '.join(extra)})"]
