"""Every user-facing panel, plus the pieces they are built from.

Four panels render a dataset (image_2d, overlay, mask_editor, spectrum): they
subclass ``components.DatasetPanel``, register with ``panel_registry``, and are
created from the Add menu. Three show and drive application state instead
(acquisition, devices, data_library): always present, built once by the window.
"""

from __future__ import annotations

from .acquisition.panel import AcquisitionPanel
from .components.dataset_panel import DatasetPanel, panel_registry
from .data_library.panel import DataLibraryPanel
from .devices.panel import DevicesPanel
from .image_2d.panel import Image2DPanel
from .mask_editor.panel import MaskEditorPanel
from .overlay.panel import OverlayPanel
from .spectrum.panel import SpectrumPanel

__all__ = [
    "DatasetPanel",
    "panel_registry",
    "Image2DPanel",
    "OverlayPanel",
    "MaskEditorPanel",
    "SpectrumPanel",
    "DataLibraryPanel",
    "DevicesPanel",
    "AcquisitionPanel",
]
