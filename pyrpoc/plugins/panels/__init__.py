"""Panels that each show one open program output, registered in
``data_panel_registry``. Importing this package registers them all."""

from __future__ import annotations

from pyrpoc.core.panels import data_panel_registry

from .image.panel import ImagePanel
from .mask_editor.panel import MaskEditorPanel
from .mosaic.panel import MosaicPanel
from .overlay.panel import OverlayPanel
from .spectrum.panel import SpectrumPanel

__all__ = [
    "data_panel_registry",
    "ImagePanel",
    "MaskEditorPanel",
    "MosaicPanel",
    "OverlayPanel",
    "SpectrumPanel",
]
