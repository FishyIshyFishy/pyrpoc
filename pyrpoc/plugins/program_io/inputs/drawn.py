"""The program inputs set on images: the drawn masks and the line each drives, the last pixel
clicked, and the galvo target. None is restored: each describes the sample in front of the
microscope now."""

from __future__ import annotations

from pyrpoc.core.program_input import SidecarProgramInput, declare, declare_sidecar

from .values.mask import Mask, MasksField, MasksPng
from .values.mask_line import MaskLinesField
from .values.pixel_pick import PickField
from .values.point import Point, PointField

MASKS: SidecarProgramInput[tuple[Mask, ...]] = declare_sidecar(
    "masks",
    # Shown by the lines input, which draws each mask beside its line.
    MasksField("Masks", "The masks drawn in a Mask Editor, which is where they are edited"),
    (),
    restored=False,
    file=MasksPng(),
)
MASK_LINES = declare(
    "mask_lines",
    MaskLinesField(
        "Masks",
        "Whether each drawn mask drives a digital output line during the scan, and which",
        masks=MASKS,
    ),
    (),
    restored=False,
)
# No program shows it: image panels write it, and a runner turns it into what its program reads.
PICK = declare(
    "pick", PickField("Pick", "The last pixel clicked on an image"), None, restored=False
)
GALVO_TARGET = declare(
    "galvo_target",
    PointField("Target", "Galvo position to park at. Pick it off an image, or type volts directly"),
    Point(),
    restored=False,
)
