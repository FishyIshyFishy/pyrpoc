"""Image displays: single-channel streamed, per-channel tiled, and RGB overlay."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout

from pyrpoc_next.gui.displays.base import DisplayWidget, display_registry
from pyrpoc_next.structs.keys import DisplayKey
from pyrpoc_next.structs.manifest import DisplayManifest
from pyrpoc_next.structs.parcels import ImageFrameParcel, PartialImageParcel


def normalize(channel: np.ndarray) -> np.ndarray:
    """Scale a channel to [0, 1] for display."""
    low, high = float(channel.min()), float(channel.max())
    if high <= low:
        return np.zeros_like(channel, dtype=np.float32)
    return ((channel - low) / (high - low)).astype(np.float32)


@display_registry.register
class StreamedDisplay(DisplayWidget):
    """Live single-channel image (first channel of the frame)."""

    manifest = DisplayManifest(
        key=DisplayKey.streamed, display_name="Streamed",
        accepted_parcels=(ImageFrameParcel, PartialImageParcel),
    )

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.view = pg.ImageView()
        layout.addWidget(self.view)

    def handle(self, parcel):
        self.view.setImage(np.asarray(parcel.data[0]).T, autoLevels=True)


@display_registry.register
class TiledDisplay(DisplayWidget):
    """All channels laid side by side in one image."""

    manifest = DisplayManifest(
        key=DisplayKey.tiled, display_name="Tiled",
        accepted_parcels=(ImageFrameParcel, PartialImageParcel),
    )

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.view = pg.ImageView()
        layout.addWidget(self.view)

    def handle(self, parcel):
        montage = np.concatenate([normalize(channel) for channel in parcel.data], axis=1)
        self.view.setImage(montage.T, autoLevels=False, levels=(0.0, 1.0))


@display_registry.register
class MultiChannelDisplay(DisplayWidget):
    """Up to three channels composited into an RGB overlay."""

    manifest = DisplayManifest(
        key=DisplayKey.multichannel, display_name="Multichannel Overlay",
        accepted_parcels=(ImageFrameParcel, PartialImageParcel),
    )

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.view = pg.ImageView()
        layout.addWidget(self.view)

    def handle(self, parcel):
        channels, height, width = parcel.data.shape
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        for index in range(min(channels, 3)):
            rgb[:, :, index] = normalize(parcel.data[index])
        self.view.setImage(np.transpose(rgb, (1, 0, 2)), autoLevels=False, levels=(0.0, 1.0))
