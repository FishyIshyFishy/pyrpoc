from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RPOCImageInput:
    data: np.ndarray
    channel_labels: list[str] = field(default_factory=list)
    source_id: str = ""

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError("RPOCImageInput.data must be channels-first [C, H, W]")
        if arr.shape[0] <= 0:
            raise ValueError("RPOCImageInput.data must contain at least one channel")
        if arr.shape[1] <= 0 or arr.shape[2] <= 0:
            raise ValueError("RPOCImageInput.data has invalid spatial dimensions")
        self.data = arr
        if not self.channel_labels:
            self.channel_labels = [f"Channel {i + 1}" for i in range(arr.shape[0])]
        if len(self.channel_labels) != arr.shape[0]:
            raise ValueError("channel_labels length must match channel dimension")

    @classmethod
    def from_array(
        cls,
        image_data: np.ndarray,
        channel_labels: list[str] | None = None,
        source_id: str = "",
    ) -> "RPOCImageInput":
        arr = np.asarray(image_data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            if arr.shape[0] <= 8:
                pass
            elif arr.shape[-1] <= 8:
                arr = np.moveaxis(arr, -1, 0)
            else:
                raise ValueError(
                    "3D input shape is ambiguous; expected channels-first [C,H,W] "
                    "or channels-last [H,W,C] with C <= 8"
                )
        else:
            raise ValueError("RPOC input must be 2D or 3D data")
        labels = channel_labels or [f"Channel {i + 1}" for i in range(arr.shape[0])]
        return cls(data=arr, channel_labels=labels, source_id=source_id)


@dataclass
class RPOCRoi:
    roi_id: int
    points: list[tuple[float, float]]
    threshold_low: float
    threshold_high: float
    modulation_level: float = 0.5
    active_channels: list[bool] = field(default_factory=list)


@dataclass
class RPOCEditorState:
    image_input: RPOCImageInput | None = None
    channel_visibility: list[bool] = field(default_factory=list)
    rois: list[RPOCRoi] = field(default_factory=list)
    show_rois: bool = True
    show_labels: bool = True
    next_roi_id: int = 1
