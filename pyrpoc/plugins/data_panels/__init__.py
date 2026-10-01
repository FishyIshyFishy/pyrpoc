"""Displays of one open dataset, added from the Panels menu. Each subclasses
``structs.plugins.data_panels.DataPanel`` and registers in ``data_panel_registry``; importing
this package registers them all."""

from __future__ import annotations

from pyrpoc.structs.plugins.data_panels import data_panel_registry

from .image_2d.panel import Image2DPanel
from .mask_editor.panel import MaskEditorPanel
from .mosaic.panel import MosaicPanel
from .overlay.panel import OverlayPanel
from .spectrum.panel import SpectrumPanel

__all__ = [
    "data_panel_registry",
    "Image2DPanel",
    "MaskEditorPanel",
    "MosaicPanel",
    "OverlayPanel",
    "SpectrumPanel",
]
