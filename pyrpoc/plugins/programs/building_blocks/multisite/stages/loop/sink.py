"""``LoopSink``: where a loop puts what it did, the units it ran on, and every image it took
of a unit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from pyrpoc.plugins.program_io.outputs import FrameTag, LoopEvent, Unit


class LoopSink(Protocol):
    """Told each event as it happens, every unit run on so far whenever one starts, and each
    unit's image before the event naming it."""

    def log(self, event: LoopEvent) -> None: ...

    def keep_units(self, units: tuple[Unit, ...]) -> None: ...

    def add_frame(self, image: np.ndarray, channels: Sequence[str], tag: FrameTag) -> None: ...
