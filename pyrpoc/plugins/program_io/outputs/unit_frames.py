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

from .building_blocks.tiff import append_pages, channel_paths, read_pages
from .image import ImageOutput

# The ``paths`` key of the frames' tags, beside one key per unit, routine and channel.
TAGS_KEY = "tags"


@dataclass(frozen=True)
class FrameTag:
    """Whose frame it is, in which routine, and the dish position of its top-left pixel."""

    unit: int
    routine: str
    left_um: float
    top_um: float


def stack_key(unit: int, routine: str, channel: str) -> str:
    return f"{unit}:{routine}:{channel}"


@program_output_registry.register("unit_frames")
class UnitFramesOutput(ImageOutput):
    """An ``ImageOutput`` of every frame in the order taken, frame ``i`` tagged ``tags[i]``.
    Each unit and routine's frames are their own TIFF stack,
    ``<root>_<name>_unit<id>_<routine>_<channel>.tiff``, so one opens elsewhere with the unit
    held still; the tags are appended to ``<root>_<name>_tags.jsonl``."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.tags: list[FrameTag] = []
        self.stacks: dict[tuple[int, str], dict[str, Path]] = {}

    def add_frame(self, array: np.ndarray, channels: Sequence[str], tag: FrameTag) -> None:
        with self.lock:
            self.tags.append(tag)
            self.add(array, channels)

    def write(self, root: Path, array: np.ndarray) -> None:
        tag = self.tags[len(self.arrays) - 1]
        key = (tag.unit, tag.routine)
        if key not in self.stacks:
            stem = root.with_name(f"{root.name}_{self.name}_unit{tag.unit}_{tag.routine}")
            self.stacks[key] = channel_paths(stem, self.channels)
            self.paths |= {
                stack_key(tag.unit, tag.routine, label): path
                for label, path in self.stacks[key].items()
            }
        append_pages(self.stacks[key], array)
        tags = root.with_name(f"{root.name}_{self.name}_tags.jsonl")
        with tags.open("a", encoding="utf-8") as log:
            log.write(json.dumps(asdict(tag)) + "\n")
        self.paths[TAGS_KEY] = tags

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
        pages = read_stacks(files, {(tag.unit, tag.routine) for tag in tags})
        taken = {key: 0 for key in pages}
        arrays = []
        for tag in tags:
            key = (tag.unit, tag.routine)
            arrays.append(pages[key][taken[key]])
            taken[key] += 1
        labels = [key.split(":", 2)[2] for key in files if key != TAGS_KEY]
        output.fill(arrays, list(dict.fromkeys(labels)))
        output.tags = tags
        return output


def read_stacks(
    files: dict[str, Path], keys: set[tuple[int, str]]
) -> dict[tuple[int, str], list[np.ndarray]]:
    """Each unit and routine's frames, read from its own stack."""
    stacks = {}
    for unit, routine in keys:
        prefix = f"{unit}:{routine}:"
        stacks[(unit, routine)] = read_pages(
            {key: path for key, path in files.items() if key.startswith(prefix)}
        )
    return stacks
