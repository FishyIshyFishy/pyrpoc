"""Renders datasets. Must not import run/ or programs/.

That rule is the display/acquisition separation, enforced by the import graph
rather than by discipline, and tests/test_import_rules.py checks it.
"""

from .base import View
from .registry import view_registry
from .image_2d import Image2DView
from .overlay import OverlayView
from .mask_editor import MaskEditorView

__all__ = ["View", "view_registry", "Image2DView", "OverlayView", "MaskEditorView"]
