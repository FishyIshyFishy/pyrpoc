"""``PixelPick``: a click on one pixel of an image."""

from __future__ import annotations

from dataclasses import dataclass

from .data import Image
from .program_output import ProgramOutput


@dataclass(frozen=True)
class PixelPick:
    """The program output clicked on, and the pixel's ``x`` along the fast axis and ``y``
    along the slow."""

    output: ProgramOutput[Image]
    x: int
    y: int
