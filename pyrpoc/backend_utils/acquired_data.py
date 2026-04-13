from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class DataKind(str, Enum):
    """Tag that identifies what a piece of acquired data represents.

    Modalities attach a DataKind to every AcquiredData they emit.
    Displays declare which kinds they accept via ACCEPTED_KINDS.
    The AcquisitionInterpreter uses this tag for routing — no display
    knowledge leaks into the modality layer, and no modality knowledge
    leaks into the display layer.
    """

    INTENSITY_FRAME = "intensity_frame"  # final per-frame intensity image — saved to disk
    PARTIAL_FRAME = "partial_frame"      # live in-progress frame — not saved

    @property
    def is_persistent(self) -> bool:
        """True if this kind of data should be written to storage."""
        return self in {DataKind.INTENSITY_FRAME}


@dataclass
class AcquiredData:
    """A single unit of data produced during acquisition.

    Modalities call on_data(AcquiredData(...)) for every piece of data
    they produce — this may happen zero, one, or many times per acquire_once()
    call, and with different DataKind values within the same acquisition.
    """

    data: np.ndarray
    kind: DataKind
    channel_labels: list[str] = field(default_factory=list)
