"""Universal data types shared across the whole app. No Qt, no hardware, no logic."""

from __future__ import annotations

from pyrpoc_next.structs.feedback import FeedbackEvent, Region
from pyrpoc_next.structs.keys import DisplayKey, InstrumentKey, ModalityKey, ModifierKey
from pyrpoc_next.structs.manifest import DisplayManifest, ModalityManifest, ModifierManifest
from pyrpoc_next.structs.parameters import (
    ChannelSelectionParameter,
    CheckboxParameter,
    ChoiceParameter,
    NumberParameter,
    Parameter,
    ParameterError,
    ParameterGroups,
    ParameterValue,
    PathParameter,
    TextParameter,
    coerce_parameter_values,
    flatten_parameters,
    validate_parameter_groups,
)
from pyrpoc_next.structs.parcels import (
    HistogramCubeParcel,
    ImageFrameParcel,
    ImageParcel,
    Parcel,
    PartialImageParcel,
)
from pyrpoc_next.structs.routine import ModifierSlot, Routine, RoutineBlock
from pyrpoc_next.structs.status import (
    CompatibilityIssue,
    CompatibilityReport,
    ConnectionStatus,
    IssueSeverity,
    RunStatus,
)

__all__ = [
    "FeedbackEvent",
    "Region",
    "DisplayKey",
    "InstrumentKey",
    "ModalityKey",
    "ModifierKey",
    "DisplayManifest",
    "ModalityManifest",
    "ModifierManifest",
    "ChannelSelectionParameter",
    "CheckboxParameter",
    "ChoiceParameter",
    "NumberParameter",
    "Parameter",
    "ParameterError",
    "ParameterGroups",
    "ParameterValue",
    "PathParameter",
    "TextParameter",
    "coerce_parameter_values",
    "flatten_parameters",
    "validate_parameter_groups",
    "HistogramCubeParcel",
    "ImageFrameParcel",
    "ImageParcel",
    "Parcel",
    "PartialImageParcel",
    "ModifierSlot",
    "Routine",
    "RoutineBlock",
    "CompatibilityIssue",
    "CompatibilityReport",
    "ConnectionStatus",
    "IssueSeverity",
    "RunStatus",
]
