"""``EmptyState``: a dashed placeholder box shown where a list has nothing yet."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class EmptyState(QFrame):
    """A dashed box with a title and a line saying how to fill the space."""

    def __init__(self, title: str, hint: str, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("emptyState")
        self.setStyleSheet(
            "#emptyState { background: transparent; border: 1px dashed palette(mid);"
            " border-radius: 6px; }"
            "#emptyState QLabel { background: transparent; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)
        heading = QLabel(title, self)
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail = QLabel(hint, self)
        detail.setStyleSheet("color: palette(mid);")
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(detail)
