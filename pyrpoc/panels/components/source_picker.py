"""A labelled combo box naming one open dataset, defaulting to the latest."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from pyrpoc.structs.data import Data, Dataset, Library


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
