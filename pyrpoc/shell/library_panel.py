"""The library panel: the runs currently open.

The dataset library needed a home in the UI, and the OptoControls manager
vacated a panel slot when masks became run parameters. Takes its place, keeping
the panel count at four: Acquisition, Devices, Views, Library.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.data.dataset import Dataset

from .app import Application


class LibraryPanel(QWidget):
    def __init__(self, app: Application, parent: QWidget | None = None):
        super().__init__(parent)
        self.app = app

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.empty_label = QLabel("No runs open. Acquired data appears here.", self)
        self.empty_label.setStyleSheet("color: palette(mid); font-style: italic;")
        root.addWidget(self.empty_label)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.content = QWidget(self.scroll)
        self.rows_layout = QVBoxLayout(self.content)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(4)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)

        self.app.library.subscribe(self.refresh)
        self.app.bridge.dataset_changed.connect(lambda *_: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        datasets = self.app.library.all()
        self.empty_label.setVisible(not datasets)

        for dataset in datasets:
            self.rows_layout.addWidget(self.build_row(dataset))
        self.rows_layout.addStretch(1)

    def build_row(self, dataset: Dataset) -> QWidget:
        row = QWidget(self.content)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 2, 4, 2)

        label = QLabel(
            f"{dataset.label}  -  {len(dataset)} frames  -  {dataset.spec.name}", row
        )
        label.setToolTip(f"started {dataset.provenance.started_at}")
        layout.addWidget(label, 1)

        close = QPushButton("Close", row)
        close.clicked.connect(lambda _checked=False, d=dataset: self.app.bridge.release(d))
        layout.addWidget(close)
        return row
