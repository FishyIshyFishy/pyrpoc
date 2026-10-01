"""How many frames to capture. Shared across modalities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, block, int_field


@block
@dataclass
class FrameCountGroup(Group):
    """Its own block rather than part of the scan, so a program that images
    once per position, like a mosaic, can declare the scan without it."""

    label: ClassVar[str] = "Frames"

    num_frames: int = int_field("Frames", 1, minimum=1, tooltip="Number of frames to capture")
