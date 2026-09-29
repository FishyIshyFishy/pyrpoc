"""Every user-facing panel, plus the pieces they are built from.

A panel is a widget that lives in the dock area. All seven -- image_2d,
overlay, mask_editor, spectrum, devices, data_library, acquisition -- inherit
the one ``Panel`` base class in ``base.py``: an identity, a title, and two
persistence hooks. Nothing about where a panel's content comes from, which is
what lets devices and acquisition inherit it as directly as image_2d does.

Four of the seven also render a dataset -- image_2d, overlay, mask_editor,
spectrum. They add a ``components.SourcePicker`` to their own layout (a combo
box naming which open dataset to show, defaulting to "Latest"), register with
``panel_registry``, and are added and removed through the window's Add menu.
They must not import run/ or programs/.

The other three -- devices, data_library, acquisition -- show and drive
application state instead of a dataset, so they have no source to pick and
add no ``SourcePicker``. They are always present, built once by
shell/window.py rather than chosen from a menu, and do not register with
``panel_registry``. acquisition in particular reaches into shell/ for the
program catalog and the run bridge, which the dataset-rendering four do not
need to.

That the fixed three and the added four are, in the abstract, all just panels
in the dock -- and differ only in how they come to exist -- is exactly why
they live together here rather than three of them staying in shell/, and why
``Panel`` carries only what every one of the seven actually needs.

components/ holds the Qt building blocks more than one panel is built from:
cards, the range slider, the source picker, the list table, the channel
colour palette. Nothing there may import a panel.
"""

from pyrpoc.structs.panel import Panel, panel_registry
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
