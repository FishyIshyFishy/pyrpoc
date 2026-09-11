"""Renders datasets. Must not import run/ or programs/.

That rule is the display/acquisition separation, enforced by the import graph
rather than by discipline: views/ may import core/ and data/, nothing else.

A view reports interaction in its own vocabulary -- ``point_picked`` carries a
dataset id and a pixel, never a voltage -- which is what lets a click on an
image start an acquisition without any view learning what hardware is.
"""

from .base import View
from .registry import view_registry
from .image_2d import Image2DView
from .overlay import OverlayView
from .mask_editor import MaskEditorView
from .spectrum import SpectrumView

__all__ = [
    "View",
    "view_registry",
    "Image2DView",
    "OverlayView",
    "MaskEditorView",
    "SpectrumView",
]
