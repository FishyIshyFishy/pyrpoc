"""The mask modifier: a region that a scanning modality turns into a TTL gate.

Pure config: the mask array plus which DAQ line to pulse. How the mask becomes a
signal is the modality's job (TTL for real scans, a pixel boost for the simulator).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from attrs import define, field

from pyrpoc_next.acquisition.modifiers.base import Modifier, modifier_registry
from pyrpoc_next.structs.keys import ModifierKey
from pyrpoc_next.structs.manifest import ModifierManifest
from pyrpoc_next.structs.parameters import NumberParameter


def mask_parameters():
    return {
        "daq": [
            NumberParameter(label="DAQ Port", default=0, minimum=0, number_type=int),
            NumberParameter(label="DAQ Line", default=0, minimum=0, number_type=int),
        ]
    }


@modifier_registry.register
@define
class MaskModifier(Modifier):
    """A binary region plus the digital line it drives during a scan."""

    key: ClassVar[ModifierKey] = ModifierKey.mask
    manifest: ClassVar[ModifierManifest] = ModifierManifest(
        key=ModifierKey.mask, display_name="Mask", parameter_groups=mask_parameters()
    )

    daq_port: int = 0
    daq_line: int = 0
    mask: np.ndarray | None = field(default=None)

    @classmethod
    def from_values(cls, values: dict[str, Any]) -> "MaskModifier":
        return cls(daq_port=int(values["DAQ Port"]), daq_line=int(values["DAQ Line"]))
