"""Every user-facing panel, plus the pieces they are built from.

A panel is a widget that lives in the dock area. Seven exist today, and they
come in two kinds:

Four render a dataset -- image_2d, overlay, mask_editor, spectrum. They share
the ``Panel`` base class in ``base.py``, register with ``panel_registry``, and
are added and removed through the window's Add menu. They must not import
run/ or programs/.

Three show and drive application state instead of a dataset -- devices,
data_library, acquisition -- and are always present, built once by
shell/window.py rather than chosen from a menu. They are not (yet) ``Panel``
subclasses and do not register with ``panel_registry``: what they show has no
"latest matching dataset" to follow, so ``Panel``'s source picker does not fit
them as written. acquisition in particular reaches into shell/ for the
program catalog and the run bridge, which the dataset-rendering four do not
need to.

That the fixed three and the added four are, in the abstract, all just panels
in the dock -- and differ only in how they come to exist -- is exactly why
they live together here rather than three of them staying in shell/. Folding
them into one base class and one lifecycle is future work, not done here.

components/ holds the Qt building blocks more than one panel is built from:
cards, the range slider, the list table, the channel colour palette. Nothing
there may import a panel.
"""

from .base import Panel
from .registry import panel_registry
from .image_2d.panel import Image2DPanel
from .overlay.panel import OverlayPanel
from .mask_editor.panel import MaskEditorPanel
from .spectrum.panel import SpectrumPanel
from .data_library.panel import DataLibraryPanel
from .devices.panel import DevicesPanel
from .acquisition.panel import AcquisitionPanel

__all__ = [
    "Panel",
    "panel_registry",
    "Image2DPanel",
    "OverlayPanel",
    "MaskEditorPanel",
    "SpectrumPanel",
    "DataLibraryPanel",
    "DevicesPanel",
    "AcquisitionPanel",
]
