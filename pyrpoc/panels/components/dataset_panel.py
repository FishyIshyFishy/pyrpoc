"""The panels the Add menu creates: each renders one open dataset."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from pyrpoc.structs.data import Data, Dataset, Library
from pyrpoc.structs.panel import Panel
from pyrpoc.structs.picks import Pick
from pyrpoc.structs.registries import Registry

from .source_picker import SourcePicker


class DatasetPanel(Panel):
    """A source picker above a ``body`` the subclass fills. ``refresh`` runs
    whenever the chosen dataset or its frames change, and hands the newest
    frame to ``show_frame``, which updates the widgets it already has.

    A chosen dataset with no frames yet, as when a run has just started,
    leaves the last frame on screen: the next run replaces the picture rather
    than blanking it, and levels and layout carry over. Only having no
    dataset at all calls ``clear``.

    Every one can be told a pick is wanted and can emit one; panels that
    cannot provide picks keep the no-op, so the host routes to all of them.
    """

    renders: ClassVar[list[type[Data]]]

    picked = pyqtSignal(object)

    def __init__(self, library: Library):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        self.body = QWidget(self)
        self.source = SourcePicker(self.renders, library, self)
        root.addWidget(self.source)
        root.addWidget(self.body, 1)
        self.shown_dataset: Dataset | None = None
        self.shown_frame: np.ndarray | None = None

    def connect_source(self) -> None:
        """Start redrawing on source changes. Called by subclasses once their
        widgets exist, so a refresh never meets a half-built panel."""
        self.source.changed.connect(self.refresh)
        self.refresh()

    def detach(self) -> None:
        """Stop following the library; called when the panel is removed."""
        self.library.unsubscribe(self.source.refresh_sources)

    @property
    def library(self) -> Library:
        return self.source.library

    def dataset(self) -> Dataset | None:
        return self.source.current()

    def refresh(self) -> None:
        dataset = self.dataset()
        if dataset is None:
            self.shown_dataset = self.shown_frame = None
            self.clear()
            return
        frame = dataset.latest()
        # Library changes re-announce the source even when nothing new arrived.
        if frame is None or (dataset is self.shown_dataset and frame is self.shown_frame):
            return
        self.shown_dataset, self.shown_frame = dataset, frame
        self.show_frame(dataset, frame)

    def show_frame(self, dataset: Dataset, frame: np.ndarray) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def set_pick_mode(self, kind: type[Pick] | None) -> None:
        """Offer picks of ``kind`` to the user, or stop offering with None."""
        del kind


panel_registry: Registry[DatasetPanel] = Registry("PanelRegistry", DatasetPanel)
