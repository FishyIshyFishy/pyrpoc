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
    """A source picker above a ``body`` the subclass fills, redrawn by
    ``refresh`` whenever the chosen dataset or its frames change.

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

    def latest(self) -> np.ndarray | None:
        dataset = self.dataset()
        return dataset.latest() if dataset is not None else None

    def refresh(self) -> None:
        raise NotImplementedError

    def set_pick_mode(self, kind: type[Pick] | None) -> None:
        """Offer picks of ``kind`` to the user, or stop offering with None."""
        del kind


panel_registry: Registry[DatasetPanel] = Registry("PanelRegistry", DatasetPanel)
