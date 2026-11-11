"""``SurveyOutput``: a mosaic whose tiles were focused where they held content, with the focus
plane, how fast things moved, where stitching placed each tile, and the overview made of
them."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording

from .building_blocks.tiff import append_pages, channel_paths, read_pages
from .mosaic import MosaicOutput, MosaicTile

# The ``paths`` keys of the survey's facts and of each overview channel, beside the mosaic's.
SURVEY_KEY = "survey"
OVERVIEW_PREFIX = "overview:"


@dataclass(frozen=True)
class Placement:
    """Where an image lies on the dish: its top-left pixel's position and its pixel size,
    in µm."""

    left_um: float
    top_um: float
    um_per_px: float


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
    """A ``MosaicOutput`` of dish-oriented tiles ``um_per_px`` µm a pixel, with how each was
    taken and, once finished, what the survey found and its overview. The facts are saved as
    ``<root>_<name>_survey.json`` after every tile, the overview as
    ``<root>_<name>_overview_<channel>.tiff``."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.um_per_px = 0.0
        self.taken: list[SurveyTile] = []
        self.finding: SurveyFinding | None = None
        self.overview_planes: np.ndarray | None = None
        self.placement: Placement | None = None

    def plan_survey(self, tiles: tuple[MosaicTile, ...], um_per_px: float) -> None:
        with self.lock:
            self.um_per_px = um_per_px
        self.plan(tiles)

    def add_tile(self, array: np.ndarray, channels: Sequence[str], taken: SurveyTile) -> None:
        with self.lock:
            self.taken.append(taken)
            self.write_survey()
        self.add(array, channels)

    def finish(self, finding: SurveyFinding, overview: np.ndarray, placement: Placement) -> None:
        """Take what the survey found, and its ``(C, H, W)`` overview lying at ``placement``."""
        planes = np.ascontiguousarray(overview, dtype=np.float32)
        with self.lock:
            self.overview_planes, self.placement, self.finding = planes, placement, finding
            if self.root is not None:
                self.write_overview(self.root, planes)
            self.write_survey()
        self.changed()

    def found(self) -> SurveyFinding:
        """What the survey found, which a finished survey always holds."""
        if self.finding is None:
            raise ValueError(f"{self.name} was not finished, so it found nothing to start from")
        return self.finding

    def overview(self) -> tuple[np.ndarray, Placement]:
        """The overview and where it lies, which a finished survey always holds."""
        with self.lock:
            if self.overview_planes is None or self.placement is None:
                raise ValueError(f"{self.name} was not finished, so it has no overview")
            return self.overview_planes, self.placement

    def write_overview(self, root: Path, planes: np.ndarray) -> None:
        stem = root.with_name(f"{root.name}_{self.name}_overview")
        files = channel_paths(stem, self.channels)
        append_pages(files, planes)
        self.paths |= {f"{OVERVIEW_PREFIX}{label}": path for label, path in files.items()}

    @property
    def field_pixels(self) -> int:
        """The side of one field, in pixels: one tile's width."""
        return self[0].shape[-1]

    @property
    def field_um(self) -> float:
        """The side of one field, in µm."""
        return self.field_pixels * self.um_per_px

    @property
    def nbytes(self) -> int:
        extra = 0 if self.overview_planes is None else self.overview_planes.nbytes
        return super().nbytes + extra

    def write_survey(self) -> None:
        if self.root is None:
            return
        path = self.root.with_name(f"{self.root.name}_{self.name}_survey.json")
        facts = {
            "um_per_px": self.um_per_px,
            "taken": [asdict(each) for each in self.taken],
            "finding": None if self.finding is None else asdict(self.finding),
            "placement": None if self.placement is None else asdict(self.placement),
        }
        path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
        self.paths[SURVEY_KEY] = path

    @classmethod
    def load(cls, *, name: str, recording: Recording, files: dict[str, Path]) -> SurveyOutput:
        overview = {
            key.removeprefix(OVERVIEW_PREFIX): path
            for key, path in files.items()
            if key.startswith(OVERVIEW_PREFIX)
        }
        mosaic_files = {
            key: path
            for key, path in files.items()
            if key != SURVEY_KEY and not key.startswith(OVERVIEW_PREFIX)
        }
        output = super().load(name=name, recording=recording, files=mosaic_files)
        facts = json.loads(files[SURVEY_KEY].read_text(encoding="utf-8"))
        output.um_per_px = float(facts["um_per_px"])
        output.taken = [
            SurveyTile(float(t["z_um"]), float(t["time_s"]), bool(t["content"]))
            for t in facts["taken"]
        ]
        if facts["finding"] is not None:
            output.finding = finding_from(facts["finding"])
        if overview:
            (output.overview_planes,) = read_pages(overview)
            output.placement = Placement(**facts["placement"])
        return output


def finding_from(raw: dict[str, Any]) -> SurveyFinding:
    motion = raw["motion"]
    # Surveys saved before focus drift was measured hold none.
    drift = raw.get("drift")
    return SurveyFinding(
        FocusPlane(**raw["focus"]),
        None if drift is None else FocusDrift(**drift),
        None if motion is None else MotionRate(**motion),
        tuple((float(left), float(top)) for left, top in raw["placed"]),
    )
