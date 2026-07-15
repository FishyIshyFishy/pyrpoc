from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DataKind(str, Enum):
    """Tag that identifies what a piece of acquired data represents.

    Sources/handlers attach a DataKind to every AcquiredData they emit.
    Displays declare which kinds they accept via accepted_kinds.
    The AcquisitionInterpreter uses this tag for routing — no display
    knowledge leaks into the acquisition layer, and no acquisition knowledge
    leaks into the display layer.
    """

    INTENSITY_FRAME = "intensity_frame"          # final per-frame intensity image — saved to disk
    PARTIAL_FRAME = "partial_frame"              # live in-progress frame — not saved
    FLIM_RAW_FRAME = "flim_raw_frame"            # (H, W, n_bins) float32 per-pixel decay histograms
    FLIM_PARTIAL_HISTOGRAM = "flim_partial_histogram"  # 1D array of photon counts per histogram bin

    @property
    def is_persistent(self) -> bool:
        """True if this kind of data should be written to storage."""
        return self in {DataKind.INTENSITY_FRAME}


@dataclass
class AcquiredData:
    """A single unit of data produced during acquisition.

    A handler returns a list of AcquiredData for each command it runs — this
    may be zero, one, or many items, with different DataKind values within the
    same command (e.g. FLIM emits an intensity frame and a raw histogram cube).
    """

    data: np.ndarray
    kind: DataKind
    channel_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
