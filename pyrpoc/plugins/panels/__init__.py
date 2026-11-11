"""Panels that each show one open program output, registered in
``data_panel_registry``. Importing this package registers them all."""

from __future__ import annotations

from pyrpoc.utils.widgets.data_panel import data_panel_registry

from .hyperspectral.panel import HyperspectralPanel
from .image.panel import ImagePanel
from .mask_editor.panel import MaskEditorPanel
from .mosaic.panel import MosaicPanel
from .multisite.panel import MultisitePanel
from .overlay.panel import OverlayPanel
from .spectrum.panel import SpectrumPanel

__all__ = [
    "data_panel_registry",
    "HyperspectralPanel",
    "ImagePanel",
    "MaskEditorPanel",
    "MosaicPanel",
    "MultisitePanel",
    "OverlayPanel",
    "SpectrumPanel",
]
