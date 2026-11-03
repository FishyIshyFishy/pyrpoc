"""Every parameter block a program can declare, one per file. A block whose
values are not plain numbers or text defines its field type and editor widget
in the same file. Pick from here when building a program. Importing this
package registers every block, so a session can name any of them.
"""

from __future__ import annotations

from .camera import CameraGroup
from .ccd_readout import CcdReadoutGroup
from .daq import DaqGroup
from .focus_search import FocusSearchGroup
from .frame_count import FrameCountGroup
from .histogram import HistogramGroup
from .modulation import MaskBinding, MaskRef, MasksField, ModulationGroup
from .mosaic import MosaicGroup
from .point import Point, PointField, PointGroup
from .roi import RoiGroup
from .scan import ScanGroup
from .split import SplitGroup
from .trigger import TriggerGroup

__all__ = [
    "CameraGroup",
    "CcdReadoutGroup",
    "DaqGroup",
    "FocusSearchGroup",
    "FrameCountGroup",
    "HistogramGroup",
    "MaskBinding",
    "MaskRef",
    "MasksField",
    "ModulationGroup",
    "MosaicGroup",
    "Point",
    "PointField",
    "PointGroup",
    "RoiGroup",
    "ScanGroup",
    "SplitGroup",
    "TriggerGroup",
]
