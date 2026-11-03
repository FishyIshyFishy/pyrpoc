"""``MaskCell``: the editor for one mask input, which takes its mask from a Mask Editor."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from pyrpoc.core.input_requests import InputRequest, InputRequests
from pyrpoc.core.mask import Mask

WAITING = "waiting for Send in a Mask Editor"


@dataclass
class PendingTake:
    """The mask request a cell has open, and whether the cell still exists."""

    request: InputRequest[Mask] | None = None
    alive: bool = True


class MaskCell(QWidget):
    """What one mask input holds, a button that asks the Mask Editors for a mask,
    and Clear, which leaves the input empty."""

    changed = pyqtSignal()

    def __init__(self, requests: InputRequests, empty_text: str, parent: QWidget):
        super().__init__(parent)
        self.requests = requests
        self.empty_text = empty_text
        self.mask: Mask | None = None
        self.pending = PendingTake()

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self)
        self.take_btn = QPushButton(self)
        self.take_btn.clicked.connect(self.on_take_clicked)
        clear_btn = QPushButton("Clear", self)
        clear_btn.setToolTip("Leave this input without a mask")
        clear_btn.clicked.connect(self.on_clear_clicked)
        row.addWidget(self.label, 1)
        row.addWidget(self.take_btn)
        row.addWidget(clear_btn)
        # A form is rebuilt on every program change, so a cell can go while its
        # request is open; the request must go with it.
        pending, requests_ = self.pending, requests
        self.destroyed.connect(lambda *_: close_pending(pending, requests_))
        self.refresh()

    @property
    def waiting(self) -> bool:
        return self.pending.request is not None and self.pending.request.open

    def value(self) -> Mask | None:
        return self.mask

    def set_value(self, mask: Mask | None) -> None:
        self.mask = mask
        self.refresh()

    def summary(self) -> str:
        return self.empty_text if self.mask is None else self.mask.describe()

    def on_take_clicked(self) -> None:
        """Ask the Mask Editors for a mask, or stop asking."""
        if self.waiting and self.pending.request is not None:
            self.requests.cancel(self.pending.request)
            return
        self.pending.request = self.requests.request(Mask, WAITING, self.on_answer)
        self.refresh()

    def on_answer(self, mask: Mask | None) -> None:
        if not self.pending.alive:
            return
        self.pending.request = None
        if mask is not None:
            self.mask = mask
            self.changed.emit()
        self.refresh()

    def on_clear_clicked(self) -> None:
        self.mask = None
        self.refresh()
        self.changed.emit()

    def refresh(self) -> None:
        if self.waiting:
            self.label.setText(WAITING + "…")
            self.take_btn.setText("Cancel")
            self.take_btn.setToolTip("Stop waiting for a mask")
        else:
            self.label.setText(self.summary())
            self.take_btn.setText("Take from Mask Editor")
            self.take_btn.setToolTip("Then press Send in a Mask Editor to put its mask here")


def close_pending(pending: PendingTake, requests: InputRequests) -> None:
    """Withdraw a gone cell's open request, without its answer reaching the cell."""
    pending.alive = False
    if pending.request is not None:
        requests.cancel(pending.request)
