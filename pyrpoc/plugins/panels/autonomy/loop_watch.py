"""``LoopWatch``: what a loop on the shown survey is doing, its latest activity, and the buttons
that steer the chosen site."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget


class LoopWatch(QWidget):
    """Shows how long a loop has run and what it did, and asks for the chosen site to be
    visited next or paused."""

    visit_next = pyqtSignal()
    pause = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.heading = QLabel("", self)
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)
        self.hint = QLabel(
            "Click a site on the map or in the list to pause it or visit it next.", self
        )
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.buttons = QWidget(self)
        row = QHBoxLayout(self.buttons)
        row.setContentsMargins(0, 0, 0, 0)
        self.pause_button = QPushButton("Pause", self.buttons)
        self.pause_button.setToolTip("Skip this site until it is resumed")
        self.pause_button.clicked.connect(self.pause.emit)
        self.visit_button = QPushButton("Visit next", self.buttons)
        self.visit_button.setToolTip("Visit this site next, ahead of the others")
        self.visit_button.clicked.connect(self.visit_next.emit)
        row.addWidget(self.pause_button)
        row.addWidget(self.visit_button)
        layout.addWidget(self.buttons)
        layout.addWidget(QLabel("Activity", self))
        self.activity = QListWidget(self)
        layout.addWidget(self.activity, 1)

    def show_heading(self, text: str) -> None:
        self.heading.setText(text)

    def show_activity(self, lines: list[str]) -> None:
        self.activity.clear()
        self.activity.addItems(lines)

    def set_site(self, number: int | None, paused: bool, running: bool) -> None:
        """Offer the buttons for the chosen site while the loop runs, or say how to choose one."""
        self.hint.setVisible(running and number is None)
        self.buttons.setVisible(running and number is not None)
        if number is not None:
            self.pause_button.setText(f"Resume site {number}" if paused else f"Pause site {number}")
            self.visit_button.setText(f"Visit site {number} next")
            self.visit_button.setEnabled(not paused)
