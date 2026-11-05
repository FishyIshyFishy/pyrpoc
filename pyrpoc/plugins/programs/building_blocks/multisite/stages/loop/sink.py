"""``LoopSink``: where a loop puts what it did, its units, and every image it took of a unit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from pyrpoc.plugins.program_io.outputs import FrameTag, LoopEvent, UnitRecord


class LoopSink(Protocol):
    """Told each event as it happens, every unit the loop has had whenever one joins or
    changes, and each unit's image before the event naming it."""

    def log(self, event: LoopEvent) -> None: ...

    def keep_units(self, records: tuple[UnitRecord, ...]) -> None: ...

    def add_frame(self, image: np.ndarray, channels: Sequence[str], tag: FrameTag) -> None: ...
