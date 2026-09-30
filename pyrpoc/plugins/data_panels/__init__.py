"""Displays of one open dataset, added from the Panels menu. Each subclasses
``structs.panel.DataPanel`` and registers in ``data_panel_registry``; importing
this package registers them all."""

from __future__ import annotations

from pyrpoc.structs.panel import data_panel_registry

from .image_2d.panel import Image2DPanel
from .mask_editor.panel import MaskEditorPanel
from .overlay.panel import OverlayPanel
from .spectrum.panel import SpectrumPanel

__all__ = [
    "data_panel_registry",
    "Image2DPanel",
    "MaskEditorPanel",
    "OverlayPanel",
    "SpectrumPanel",
]
