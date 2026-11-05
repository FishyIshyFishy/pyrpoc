"""``CleanFolder``: a pipeline step starts only in a folder holding just the earlier steps it
reads."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core.check import Check, CheckContext


@dataclass(frozen=True)
class CleanFolder(Check):
    """Refuses a folder holding any recording the program does not read. Each read then comes
    from the only recording that can supply it, and a step with no reads needs an empty
    folder."""

    def blockers(self, ctx: CheckContext) -> list[str]:
        read = set(ctx.reading().values())
        # Failed and unreadable recordings count too: their files are in the folder all the same.
        extra = [name for name in ctx.recordings_in_folder() if name not in read]
        if not extra:
            return []
        return [f"a folder holding only what it reads (move out {', '.join(extra)})"]
