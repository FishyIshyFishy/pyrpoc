"""Temporary: push dataset frames at the v3.0 displays.

Deliberate throwaway, deleted in phase 8. Its only job is to keep the two
existing displays working while acquisition changes underneath them, so this
commit leaves an application that runs.

The displays still expect ``render(AcquiredData)`` and still own their arrays.
Phase 8 replaces them with views that read from a bound dataset, at which point
this file and the whole push model go.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.core.streams import Image2D
from pyrpoc.data.dataset import Dataset

from .app import Application


class DisplayBridge(QObject):
    """Renders Image2D frames into whichever v3.0 displays are open."""

    def __init__(self, app: Application, parent: QObject | None = None):
        super().__init__(parent)
        self.app = app
        app.bridge.dataset_changed.connect(self.on_dataset_changed)

    def on_dataset_changed(self, dataset: Dataset, index: int) -> None:
        if dataset.spec is not Image2D:
            return  # Cube3D and Samples4D have no v3.0 display
        frame = dataset.frame(index)
        acquired = AcquiredData(
            data=frame,
            kind=DataKind.INTENSITY_FRAME,
            channel_labels=list(dataset.channel_labels),
            metadata=dict(dataset.metadata),
        )
        for display in list(self.app.views):
            if not getattr(display, "attached", True):
                continue
            if not getattr(display, "docked_visible", True):
                continue
            if DataKind.INTENSITY_FRAME not in getattr(display, "accepted_kinds", []):
                continue
            try:
                display.render(acquired)
            except Exception as exc:  # noqa: BLE001 - one bad display must not stop a run
                display.last_error = str(exc)
