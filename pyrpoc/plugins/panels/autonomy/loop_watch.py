"""``LoopWatch``: what a running loop is doing beside the units, its queue and its latest
events, and the buttons that steer one unit."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

from pyrpoc.plugins.program_io.outputs import LoopView


class LoopWatch(QWidget):
    """Shows a loop's queue and events, and asks for a unit to run now or to pause."""

    run_now = pyqtSignal()
    pause = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.heading = QLabel("", self)
        layout.addWidget(self.heading)
        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run now", self)
        self.run_button.setToolTip("Scan the chosen unit next, ahead of everything due")
        self.run_button.clicked.connect(self.run_now.emit)
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause.emit)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.pause_button)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Queue, overdue first", self))
        self.queue = QListWidget(self)
        layout.addWidget(self.queue, 1)
        layout.addWidget(QLabel("Recent", self))
        self.events = QListWidget(self)
        layout.addWidget(self.events, 1)

    def show_view(self, view: LoopView, running: bool) -> None:
        verb = "running" if running else "finished"
        self.heading.setText(f"Loop {verb}, {view.elapsed_s / 60.0:.1f} min")
        self.queue.clear()
        for entry in view.queue:
            wait = entry.due_s - view.elapsed_s
            when = f"{-wait:.0f} s overdue" if wait < 0 else f"in {wait:.0f} s"
            self.queue.addItem(f"unit {entry.unit} · {entry.routine} · {when}")
        self.events.clear()
        self.events.addItems(list(view.recent))

    def set_unit(self, chosen: bool, paused: bool, running: bool, run_now_taken: bool) -> None:
        """Enable the buttons for a chosen unit of a running loop."""
        self.run_button.setEnabled(running and chosen and not paused and run_now_taken)
        self.pause_button.setEnabled(running and chosen)
        self.pause_button.setText("Resume" if paused else "Pause")
