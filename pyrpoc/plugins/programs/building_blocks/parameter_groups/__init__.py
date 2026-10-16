"""Every parameter block a program can declare, one per file. A block whose
values are not plain numbers or text defines its field type and editor widget
in the same file. Pick from here when building a program. Importing this
package registers every block, so a session can name any of them.
"""

from __future__ import annotations

from .daq import DaqGroup
from .focus_search import FocusSearchGroup
from .frame import FrameGroup
from .frame_count import FrameCountGroup
from .histogram import HistogramGroup
from .modulation import Mask, MaskRef, MasksField, ModulationGroup
from .mosaic import MosaicGroup, Tile
from .pacing import PacingGroup
from .point import Point, PointField, PointGroup
from .roi import RoiGroup
from .scan import ScanGroup
from .signal import PATTERNS, SignalGroup
from .specimen import SpecimenGroup
from .spectrum import SpectrumGroup
from .split import SplitGroup
from .trigger import TriggerGroup

__all__ = [
    "DaqGroup",
    "FocusSearchGroup",
    "FrameCountGroup",
    "FrameGroup",
    "HistogramGroup",
    "Mask",
    "MaskRef",
    "MasksField",
    "ModulationGroup",
    "MosaicGroup",
    "PacingGroup",
    "PATTERNS",
    "Point",
    "PointField",
    "PointGroup",
    "RoiGroup",
    "ScanGroup",
    "SignalGroup",
    "SpecimenGroup",
    "SpectrumGroup",
    "SplitGroup",
    "Tile",
    "TriggerGroup",
]
