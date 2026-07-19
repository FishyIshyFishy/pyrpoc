"""Data storage: complete frames to multi-page TIFF per channel, histograms to NPZ.

Only complete ImageFrameParcels are written; partial (streaming) frames never are —
which falls out of the parcel type, no flag needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyrpoc_next.structs.parcels import HistogramCubeParcel, ImageFrameParcel, Parcel


class FrameStorage:
    """Accumulates a run's output to disk under a root path."""

    def __init__(self):
        self.root: Path | None = None
        self.writers: dict[str, object] = {}
        self.histograms: list[np.ndarray] = []

    def begin(self, root: str | Path) -> None:
        """Start a run: prepare the output directory rooted at ``root``."""
        self.root = Path(root).expanduser()
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.writers = {}
        self.histograms = []

    def save(self, parcel: Parcel) -> None:
        """Persist a parcel if it is a kind we store."""
        if self.root is None:
            return
        if isinstance(parcel, ImageFrameParcel):
            self.save_image(parcel)
        elif isinstance(parcel, HistogramCubeParcel):
            self.histograms.append(parcel.data)

    def save_image(self, parcel: ImageFrameParcel) -> None:
        """Append each channel of a frame to its own multi-page TIFF."""
        import tifffile

        for label, channel in zip(parcel.channel_labels, parcel.data):
            writer = self.writers.get(label)
            if writer is None:
                writer = tifffile.TiffWriter(f"{self.root}_{label}.tiff")
                self.writers[label] = writer
            writer.write(np.asarray(channel, dtype=np.float32), contiguous=True)

    def finish(self) -> None:
        """Close writers and flush any accumulated histogram cubes to NPZ."""
        for writer in self.writers.values():
            writer.close()  # pyright: ignore
        self.writers = {}
        if self.histograms and self.root is not None:
            np.savez_compressed(f"{self.root}_histograms.npz", frames=np.stack(self.histograms))
        self.histograms = []
