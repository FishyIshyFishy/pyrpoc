"""The pieces programs are assembled from.

A program is a composition: it declares which parameter blocks it wants and
writes the loop that drives them. What it composes lives here.

May import ``structs/`` and ``devices/``. Nothing here knows which program is
using it, and nothing here writes a dataset.

``editors.py`` is the one Qt module: the widgets for the field types declared
in ``param_groups``, handed to the form through ``Field.editor``. It is imported
only when an editor is asked for, so programs import with no Qt in sight.
"""

from .param_groups import (
    DaqGroup,
    FrameGroup,
    HistogramGroup,
    Mask,
    MasksField,
    ModulationGroup,
    PacingGroup,
    PATTERNS,
    Point,
    PointField,
    PointGroup,
    ScanGroup,
    SignalGroup,
    SpectrumGroup,
    SplitGroup,
    TriggerGroup,
    masks_field,
    point_field,
)

__all__ = [
    "DaqGroup",
    "FrameGroup",
    "HistogramGroup",
    "Mask",
    "MasksField",
    "ModulationGroup",
    "PacingGroup",
    "PATTERNS",
    "Point",
    "PointField",
    "PointGroup",
    "ScanGroup",
    "SignalGroup",
    "SpectrumGroup",
    "SplitGroup",
    "TriggerGroup",
    "masks_field",
    "point_field",
]
