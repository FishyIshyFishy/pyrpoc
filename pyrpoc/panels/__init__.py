"""Every dockable screen: the three fixed panels and the addable views.

Fixed: ``acquisition.LauncherPanel``, ``devices.DevicesPanel``,
``data_library.DataPanel``, ``views_panel.ViewsPanel``. One instance each,
always present, and free to import ``pyrpoc.app`` -- they exist to call it.
Import these from their own module (``from pyrpoc.panels.devices import
DevicesPanel``) rather than from this package: ``pyrpoc.app`` imports back
into this package (``registry``, for restoring a saved session's views), so
re-exporting the app-aware panels here too would make the two packages import
each other at package-init time.

Addable, and re-exported below: ``Image2DView``, ``OverlayView``,
``MaskEditorView``, ``SpectrumView``, created on request through
``view_registry``. These may not import ``run/``, ``programs/`` or
``pyrpoc.app``, enforced by the import graph rather than by discipline: a view
may import only ``core/`` and ``data/``. That is the display/acquisition
separation -- a view reports interaction in its own vocabulary
(``point_picked`` carries a dataset id and a pixel, never a voltage), which is
what lets a click on an image start an acquisition without any view learning
what hardware is.
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
