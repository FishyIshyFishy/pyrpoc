"""Widgets that live in the dock area, and the data-panel contract.

A ``Panel`` has identity and a title. A ``DataPanel`` shows one open dataset,
chosen with a ``SourcePicker``; data-panel plugins subclass it and register in
``data_panel_registry``. The only Qt file in structs, since a panel is a widget.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .data import Data
from .dataset import Dataset
from .device import make_instance_id
from .library import Library
from .picks import Pick
from .registry import Registry


class Panel(QWidget):
    """A widget that lives in the dock area."""

    display_name: str = "Panel"
    registry_key: str = "panel"

    def __init__(self) -> None:
        super().__init__()
        self.instance_id = make_instance_id(self.registry_key)
        self.user_label: str | None = None

    @property
    def type_key(self) -> str:
        return self.registry_key

    @property
    def title(self) -> str:
        return self.user_label or self.display_name

    def export_persistence_state(self) -> dict[str, Any]:
        return {}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        del state


class SourcePicker(QWidget):
    """Names one dataset matching ``renders``, out of everything open. "Latest"
    follows the newest match, so a new run shows up without a click."""

    # The chosen dataset may have changed; the panel re-reads ``current()``.
    changed = pyqtSignal()

    def __init__(self, renders: Sequence[type[Data]], library: Library, parent: QWidget):
        super().__init__(parent)
        self.renders = list(renders)
        self.library = library
        self._follow_latest = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Source:", self))
        self.combo = QComboBox(self)
        self.combo.currentIndexChanged.connect(self.on_chosen)
        layout.addWidget(self.combo, 1)

        library.subscribe(self.refresh_sources)
        self.refresh_sources()

    def refresh_sources(self) -> None:
        """Rebuild the picker, keeping the current choice where possible."""
        current = self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("Latest", None)
        for dataset in self.library.matching(*self.renders):
            self.combo.addItem(dataset.label, dataset.id)
        if not self._follow_latest:
            self.combo.setCurrentIndex(max(self.combo.findData(current), 0))
        self.combo.blockSignals(False)
        self.changed.emit()

    def on_chosen(self, _index: int) -> None:
        self._follow_latest = self.combo.currentData() is None
        self.changed.emit()

    def current(self) -> Dataset | None:
        chosen = self.combo.currentData()
        if chosen is None:
            candidates = self.library.matching(*self.renders)
            return candidates[0] if candidates else None
        return self.library.by_id(chosen)


class DataPanel(Panel):
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


data_panel_registry: Registry[DataPanel] = Registry("DataPanelRegistry", DataPanel)
