"""A labelled combo box naming one open dataset, defaulting to the latest.

Pulled out of the panel base class: a source picker is not a property of
"being a panel", it is a property of showing a dataset that outlives its
renderer, which only the four dataset-rendering panels do. Each of them adds
one of these to its own layout and reads ``current()`` after ``changed`` --
composition, not another layer of ``Panel`` to inherit.

"Latest" follows the newest matching dataset, which reproduces v3.0's
implicit behaviour of pushing the current run at whatever was open.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from pyrpoc.src.structs.data import Data

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.src.app.library import DataLibrary
    from pyrpoc.src.structs.data import Dataset


class SourcePicker(QWidget):
    """Names one dataset matching ``renders``, out of everything open."""

    # The chosen dataset may have changed -- library membership moved, or the
    # user picked a different entry. The panel re-reads ``current()``.
    changed = pyqtSignal()

    def __init__(self, renders: Sequence[type[Data]], parent: QWidget | None = None):
        super().__init__(parent)
        self.renders = list(renders)
        self._library: DataLibrary | None = None
        self._follow_latest = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Source:", self))
        self.combo = QComboBox(self)
        self.combo.currentIndexChanged.connect(self._on_chosen)
        layout.addWidget(self.combo, 1)

    # -- library --------------------------------------------------------------- #

    def attach_library(self, library: DataLibrary) -> None:
        self._library = library
        library.subscribe(self.refresh_sources)
        self.refresh_sources()

    def library(self) -> DataLibrary | None:
        return self._library

    def candidates(self) -> list[Dataset]:
        if self._library is None:
            return []
        return self._library.matching(*self.renders)

    def refresh_sources(self) -> None:
        """Rebuild the picker, keeping the current choice where possible."""
        current = self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("Latest", None)
        for dataset in self.candidates():
            self.combo.addItem(dataset.label, dataset.id)
        if not self._follow_latest and isinstance(current, str):
            index = self.combo.findData(current)
            self.combo.setCurrentIndex(max(index, 0))
        self.combo.blockSignals(False)
        self.changed.emit()

    def _on_chosen(self, _index: int) -> None:
        self._follow_latest = self.combo.currentData() is None
        self.changed.emit()

    # -- the chosen dataset ------------------------------------------------------ #

    def current(self) -> Dataset | None:
        chosen = self.combo.currentData()
        if chosen is None:
            candidates = self.candidates()
            return candidates[0] if candidates else None
        if self._library is not None:
            return self._library.by_id(chosen)
        return None
