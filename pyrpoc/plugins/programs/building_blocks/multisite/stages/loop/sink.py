"""``LoopSink``: where a loop puts what it did and every image it took of a unit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from pyrpoc.plugins.program_io.outputs import FrameTag, LoopEvent


class LoopSink(Protocol):
    """Told each event as it happens, and given each unit's image before the event naming it."""

    def log(self, event: LoopEvent) -> None: ...

    def add_frame(self, image: np.ndarray, channels: Sequence[str], tag: FrameTag) -> None: ...
