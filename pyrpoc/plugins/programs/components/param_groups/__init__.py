"""Every parameter block a program can be built from, one per file, plus the
field types that are this instrument's own. Importing this package registers
every block, so a session can name any of them.
"""

from __future__ import annotations

from .daq import DaqGroup
from .frame import FrameGroup
from .frame_count import FrameCountGroup
from .histogram import HistogramGroup
from .mask_field import Mask, MasksField, masks_field
from .modulation import ModulationGroup
from .mosaic import MosaicGroup, Tile
from .pacing import PacingGroup
from .point import PointGroup
from .point_field import Point, PointField, point_field
from .scan import ScanGroup
from .signal import PATTERNS, SignalGroup
from .specimen import SpecimenGroup
from .spectrum import SpectrumGroup
from .split import SplitGroup
from .trigger import TriggerGroup

__all__ = [
    "DaqGroup",
    "FrameCountGroup",
    "FrameGroup",
    "HistogramGroup",
    "Mask",
    "masks_field",
    "MasksField",
    "ModulationGroup",
    "MosaicGroup",
    "PacingGroup",
    "PATTERNS",
    "Point",
    "point_field",
    "PointField",
    "PointGroup",
    "ScanGroup",
    "SignalGroup",
    "SpecimenGroup",
    "SpectrumGroup",
    "SplitGroup",
    "Tile",
    "TriggerGroup",
]
