"""``SurveyLogOutput``: everything a survey did, in order, and its events, one class for each
kind. Tiles are named by their place in the walk, from 0; times are on the instrument's clock."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.core.program_output import program_output_registry

from .log import LogOutput


@dataclass(frozen=True)
class TileVisiting:
    """The stage began moving to a tile, the ``tile``-th of ``tiles``."""

    kind: ClassVar[str] = "visiting"
    time_s: float
    tile: int
    tiles: int

    def text(self) -> str:
        return f"tile {self.tile + 1} of {self.tiles}: moving to it"


@dataclass(frozen=True)
class TileFocusing:
    """A tile held content, so focus began to be searched for there."""

    kind: ClassVar[str] = "tile focusing"
    time_s: float
    tile: int
    tiles: int

    def text(self) -> str:
        return f"tile {self.tile + 1} of {self.tiles}: focusing"


@dataclass(frozen=True)
class TileTaken:
    """A tile was imaged at ``z_um``, holding content or not."""

    kind: ClassVar[str] = "taken"
    time_s: float
    tile: int
    tiles: int
    z_um: float
    content: bool

    def text(self) -> str:
        held = "" if self.content else ", nothing in it"
        return f"tile {self.tile + 1} of {self.tiles}: taken at z {self.z_um:.1f} µm{held}"


@dataclass(frozen=True)
class MotionRevisiting:
    """An early tile began to be imaged again, the ``revisit``-th of ``revisits``, to measure
    how fast things move."""

    kind: ClassVar[str] = "revisiting"
    time_s: float
    revisit: int
    revisits: int

    def text(self) -> str:
        return f"measuring motion: revisit {self.revisit + 1} of {self.revisits}"


@dataclass(frozen=True)
class MotionMeasured:
    """How fast things move, in µm per minute, from ``samples`` features; None with none."""

    kind: ClassVar[str] = "motion"
    time_s: float
    um_per_min: float | None
    samples: int

    def text(self) -> str:
        if self.um_per_min is None:
            return "motion not measured: no feature was found again"
        return f"things move {self.um_per_min:.2f} µm per minute ({self.samples} features)"


@dataclass(frozen=True)
class Stitching:
    """Placing the tiles into one overview began."""

    kind: ClassVar[str] = "stitching"
    time_s: float

    def text(self) -> str:
        return "placing the tiles into the overview"


@dataclass(frozen=True)
class SurveyEnded:
    """The overview is made and the stage is back where the survey started."""

    kind: ClassVar[str] = "ended"
    time_s: float

    def text(self) -> str:
        return "survey done, back at its start"


SurveyEvent = (
    TileVisiting
    | TileFocusing
    | TileTaken
    | MotionRevisiting
    | MotionMeasured
    | Stitching
    | SurveyEnded
)
SURVEY_EVENTS: dict[str, type[SurveyEvent]] = {
    cls.kind: cls
    for cls in (
        TileVisiting,
        TileFocusing,
        TileTaken,
        MotionRevisiting,
        MotionMeasured,
        Stitching,
        SurveyEnded,
    )
}


@program_output_registry.register("survey_log")
class SurveyLogOutput(LogOutput[SurveyEvent]):
    """A survey's events."""

    def kinds(self) -> Mapping[str, type[SurveyEvent]]:
        return SURVEY_EVENTS
