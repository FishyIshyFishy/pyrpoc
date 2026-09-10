"""The pieces programs are assembled from.

A program is a composition: it declares which parameter blocks it wants and
writes the loop that drives them. What it composes lives here.

May import ``core/`` and ``devices/``. Nothing here knows which program is
using it, and nothing here writes a dataset.
"""

from .param_groups import (
    BLOCKS,
    DaqGroup,
    FrameGroup,
    FramesGroup,
    HistogramGroup,
    Mask,
    MasksField,
    ModulationGroup,
    PacingGroup,
    PATTERNS,
    ScanGroup,
    SignalGroup,
    SplitGroup,
    TriggerGroup,
    block,
    masks_field,
)

__all__ = [
    "BLOCKS",
    "DaqGroup",
    "FrameGroup",
    "FramesGroup",
    "HistogramGroup",
    "Mask",
    "MasksField",
    "ModulationGroup",
    "PacingGroup",
    "PATTERNS",
    "ScanGroup",
    "SignalGroup",
    "SplitGroup",
    "TriggerGroup",
    "block",
    "masks_field",
]
