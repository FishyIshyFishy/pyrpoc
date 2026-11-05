"""``SurveySink``: where a survey puts what it takes and what it finds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from pyrpoc.plugins.program_io.outputs import MosaicTile, SurveyFinding, SurveyTile

from ...interfaces.overview import Overview


class SurveySink(Protocol):
    """Told the grid before the first tile, each tile as it is taken, and at the end the overview
    and what the survey found."""

    def plan(self, tiles: tuple[MosaicTile, ...], um_per_px: float) -> None: ...

    def add_tile(self, frame: np.ndarray, channels: Sequence[str], taken: SurveyTile) -> None: ...

    def finish(self, finding: SurveyFinding, overview: Overview) -> None: ...
