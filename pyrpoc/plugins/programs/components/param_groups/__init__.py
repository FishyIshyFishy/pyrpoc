"""Every parameter block a program can be built from, one per file, plus the
field types that are this instrument's own. Importing this package registers
every block, so a workspace can name any of them.
"""

from __future__ import annotations

from .daq import DaqGroup
from .frame import FrameGroup
from .histogram import HistogramGroup
from .mask_field import Mask, MasksField, masks_field
from .modulation import ModulationGroup
from .pacing import PacingGroup
from .point import PointGroup
from .point_field import Point, PointField, point_field
from .scan import ScanGroup
from .signal import PATTERNS, SignalGroup
from .spectrum import SpectrumGroup
from .split import SplitGroup
from .trigger import TriggerGroup

__all__ = [
    "DaqGroup",
    "FrameGroup",
    "HistogramGroup",
    "Mask",
    "masks_field",
    "MasksField",
    "ModulationGroup",
    "PacingGroup",
    "PATTERNS",
    "Point",
    "point_field",
    "PointField",
    "PointGroup",
    "ScanGroup",
    "SignalGroup",
    "SpectrumGroup",
    "SplitGroup",
    "TriggerGroup",
]
