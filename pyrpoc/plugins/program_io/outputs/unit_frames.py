"""``UnitFramesOutput``: the frames a loop took of its units, each centred on where its unit was
tracked to."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording

from .building_blocks.tiff import append_planes, read_planes
from .image import ImageOutput

# The ``paths`` keys of the frames' tags and channels, beside one key per unit and routine.
TAGS_KEY = "tags"
CHANNELS_KEY = "channels"


@dataclass(frozen=True)
class FrameTag:
    """Whose frame it is, in which routine, and the dish position of its top-left pixel."""

    unit: int
    routine: str
    left_um: float
    top_um: float


def stack_key(unit: int, routine: str) -> str:
    return f"{unit}:{routine}"


@program_output_registry.register("unit_frames")
class UnitFramesOutput(ImageOutput):
    """An ``ImageOutput`` of every frame in the order taken, frame ``i`` tagged ``tags[i]``.
    Each unit's frames in each routine are one file, ``<folder>/<routine>/unit<id>.tiff``, a
    page per channel of each frame, so a unit opens elsewhere held still. The tags are
    appended to ``<root>_<name>_tags.jsonl`` and the channels named in
    ``<root>_<name>_channels.json``."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.tags: list[FrameTag] = []
        self.stacks: dict[tuple[int, str], Path] = {}

    def add_frame(self, array: np.ndarray, channels: Sequence[str], tag: FrameTag) -> None:
        with self.lock:
            self.tags.append(tag)
            self.add(array, channels)

    def write(self, root: Path, array: np.ndarray) -> None:
        if not self.stacks:
            self.write_channels(root)
        tag = self.tags[len(self.arrays) - 1]
        key = (tag.unit, tag.routine)
        if key not in self.stacks:
            # Named by routine and unit alone, since a folder holds one loop.
            path = root.parent / tag.routine / f"unit{tag.unit}.tiff"
            path.parent.mkdir(exist_ok=True)
            path.unlink(missing_ok=True)
            self.stacks[key] = path
            self.paths[stack_key(*key)] = path
        append_planes(self.stacks[key], array)
        tags = root.with_name(f"{root.name}_{self.name}_tags.jsonl")
        with tags.open("a", encoding="utf-8") as log:
            log.write(json.dumps(asdict(tag)) + "\n")
        self.paths[TAGS_KEY] = tags

    def write_channels(self, root: Path) -> None:
        path = root.with_name(f"{root.name}_{self.name}_channels.json")
        path.write_text(json.dumps(list(self.channels)), encoding="utf-8")
        self.paths[CHANNELS_KEY] = path

    def of_unit(self, unit: int, routine: str) -> list[np.ndarray]:
        """The frames of one unit and routine, oldest first."""
        with self.lock:
            return [
                array
                for array, tag in zip(self.arrays, self.tags, strict=True)
                if (tag.unit, tag.routine) == (unit, routine)
            ]

    @classmethod
    def load(cls, *, name: str, recording: Recording, files: dict[str, Path]) -> UnitFramesOutput:
        output = cls(name=name, recording=recording, root=None)
        if TAGS_KEY not in files:
            return output
        lines = files[TAGS_KEY].read_text(encoding="utf-8").splitlines()
        tags = [FrameTag(**json.loads(line)) for line in lines if line]
        channels = [str(label) for label in json.loads(files[CHANNELS_KEY].read_text("utf-8"))]
        stacks = {
            key: read_planes(files[stack_key(*key)], len(channels))
            for key in {(tag.unit, tag.routine) for tag in tags}
        }
        taken = dict.fromkeys(stacks, 0)
        arrays = []
        for tag in tags:
            key = (tag.unit, tag.routine)
            arrays.append(stacks[key][taken[key]])
            taken[key] += 1
        output.fill(arrays, channels)
        output.tags = tags
        return output
