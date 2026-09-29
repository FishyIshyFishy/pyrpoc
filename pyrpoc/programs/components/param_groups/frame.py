"""The shape of a simulated frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.params import Group, int_field
from pyrpoc.structs.registries import block


@block
@dataclass
class FrameGroup(Group):
    """The shape of a simulated frame; ``ScanGroup`` decides it on a real rig."""

    label: ClassVar[str] = "Frame"

    num_frames: int = int_field("Frames", 1, minimum=1, tooltip="Number of frames to capture")
    x_pixels: int = int_field("X Pixels", 256, minimum=8, tooltip="Frame width in pixels")
    y_pixels: int = int_field("Y Pixels", 256, minimum=8, tooltip="Frame height in pixels")
    channels: int = int_field(
        "Channels", 2, minimum=1, maximum=16, tooltip="How many detector channels to fake"
    )
