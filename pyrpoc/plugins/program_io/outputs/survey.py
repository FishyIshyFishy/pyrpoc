"""``SurveyOutput``: a mosaic whose tiles were focused where they held content, with the focus
plane, how fast things moved, and where stitching placed each tile."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording

from .mosaic import MosaicOutput, Placement

# The ``paths`` key of the survey's facts, beside the mosaic's.
SURVEY_KEY = "survey"


@dataclass(frozen=True)
class FocusPlane:
    """Focus as a plane over stage position, ``z = z0 + tilt_x * x + tilt_y * y`` in µm, the
    spread of the points it was fitted to about it, and how many there were."""

    z0_um: float
    tilt_x: float
    tilt_y: float
    residual_um: float
    points: int

    def z_at(self, x_um: float, y_um: float) -> float:
        return self.z0_um + self.tilt_x * x_um + self.tilt_y * y_um


@dataclass(frozen=True)
class FocusDrift:
    """How fast focus drifted over a survey, in µm per minute, and its standard error."""

    um_per_min: float
    error_um_per_min: float


@dataclass(frozen=True)
class MotionRate:
    """How fast what was imaged moved, in µm per minute: the median over every feature
    followed, the spread about it, and how many there were."""

    um_per_min: float
    spread_um_per_min: float
    samples: int


@dataclass(frozen=True)
class SurveyTile:
    """How one tile was taken: the z, when on the instrument's clock, and whether it held
    content (and so was focused)."""

    z_um: float
    time_s: float
    content: bool


@dataclass(frozen=True)
class SurveyFinding:
    """What the whole survey found: the focus plane, as it stood when the last tile was taken,
    and how fast focus drifted (None if not measured), how fast things move (None with nothing
    to follow), and the dish position stitching gave each tile's top-left pixel."""

    focus: FocusPlane
    drift: FocusDrift | None
    motion: MotionRate | None
    placed: tuple[tuple[float, float], ...]


@program_output_registry.register("survey")
class SurveyOutput(MosaicOutput):
    """A ``MosaicOutput`` with how each tile was taken and, once finished, what the survey
    found. The facts are saved as ``<root>_<name>_survey.json`` after every tile."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.taken: list[SurveyTile] = []
        self.finding: SurveyFinding | None = None

    def add_tile(self, array: np.ndarray, channels: Sequence[str], taken: SurveyTile) -> None:
        with self.lock:
            self.taken.append(taken)
            self.write_survey()
        self.add(array, channels)

    def finish(self, finding: SurveyFinding, overview: np.ndarray, placement: Placement) -> None:
        """Take what the survey found, and its ``(C, H, W)`` overview lying at ``placement``."""
        with self.lock:
            self.finding = finding
            self.write_survey()
        self.stitched(overview, placement)

    def found(self) -> SurveyFinding:
        """What the survey found, which a finished survey always holds."""
        if self.finding is None:
            raise ValueError(f"{self.name} was not finished, so it found nothing to start from")
        return self.finding

    def write_survey(self) -> None:
        if self.root is None:
            return
        path = self.root.with_name(f"{self.root.name}_{self.name}_survey.json")
        facts = {
            "taken": [asdict(each) for each in self.taken],
            "finding": None if self.finding is None else asdict(self.finding),
        }
        path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
        self.paths[SURVEY_KEY] = path

    @classmethod
    def names_channel(cls, key: str) -> bool:
        return key != SURVEY_KEY and super().names_channel(key)

    @classmethod
    def load(cls, *, name: str, recording: Recording, files: dict[str, Path]) -> SurveyOutput:
        output = super().load(name=name, recording=recording, files=files)
        facts = json.loads(files[SURVEY_KEY].read_text(encoding="utf-8"))
        output.taken = [
            SurveyTile(float(t["z_um"]), float(t["time_s"]), bool(t["content"]))
            for t in facts["taken"]
        ]
        if facts["finding"] is not None:
            output.finding = finding_from(facts["finding"])
        return output


def finding_from(raw: dict[str, Any]) -> SurveyFinding:
    motion = raw["motion"]
    drift = raw["drift"]
    return SurveyFinding(
        FocusPlane(**raw["focus"]),
        None if drift is None else FocusDrift(**drift),
        None if motion is None else MotionRate(**motion),
        tuple((float(left), float(top)) for left, top in raw["placed"]),
    )
