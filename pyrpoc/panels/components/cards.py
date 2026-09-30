"""Collapsible cards: an expand arrow and a title, a description line shown
while collapsed, and a body shown while expanded."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

_REMOVE_BTN_COLOR = "#e75480"

# Scoped by object name so a card's frame styling does not reach into its body.
_CARD_STYLESHEET = (
    "#card {"
    "background: palette(base);"
    "border: 1px solid palette(midlight);"
    "border-radius: 6px;"
    "}"
    # ``.QWidget`` is exactly QWidget: plain containers go transparent while
    # buttons and inputs keep the theme's fill, so they stand out from the card.
    "#card .QWidget { background: transparent; }"
    "#card QLabel, #card QCheckBox, #card QToolButton { background: transparent; }"
    # The theme fills buttons with the window colour, which in light themes is
    # the card's own colour, so buttons are outlined in the accent instead.
    "#card QPushButton {"
    "background: palette(button);"
    "color: palette(button-text);"
    "border: 1px solid palette(highlight);"
    "border-radius: 4px;"
    "padding: 3px 10px;"
    "}"
    "#card QPushButton:hover, #card QPushButton:pressed {"
    "background: palette(highlight);"
    "color: palette(highlighted-text);"
    "padding: 3px 10px;"
    "}"
    "#card QPushButton:disabled {"
    "background: transparent;"
    "color: palette(mid);"
    "border: 1px solid palette(mid);"
    "}"
    "#card QCheckBox::indicator:checked {"
    "background: palette(highlight);"
    "border: 1px solid palette(highlight);"
    "color: white;"
    "}"
    "#card QCheckBox::indicator:unchecked {"
    "background: transparent;"
    "border: 1px solid palette(mid);"
    "}"
)


class BaseCardWidget(QFrame):
    """A collapsible card, starting collapsed."""

    def __init__(self, title: str, parent: QWidget) -> None:
        super().__init__(parent)
        self._expanded = False
        self.setObjectName("card")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(_CARD_STYLESHEET)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(2)

        self.header_row = QHBoxLayout()
        self.header_row.setContentsMargins(0, 0, 0, 0)
        self.header_row.setSpacing(4)
        self.expand_btn = QToolButton(self)
        self.expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.expand_btn.setAutoRaise(True)
        self.expand_btn.setToolTip("Expand")
        self.expand_btn.clicked.connect(lambda: self.set_expanded(not self._expanded))
        self.header_row.addWidget(self.expand_btn)
        self.title_label = QLabel(title, self)
        self.header_row.addWidget(self.title_label, 1)
        root.addLayout(self.header_row)

        self._description_label = QLabel("", self)
        self._description_label.setStyleSheet(
            "color: palette(mid); font-size: 9pt; padding-left: 22px;"
        )
        self._description_label.setWordWrap(True)
        self._description_label.setVisible(False)
        root.addWidget(self._description_label)

        self.body_container = QWidget(self)
        self.body_layout = QVBoxLayout(self.body_container)
        self.body_layout.setContentsMargins(0, 2, 0, 0)
        self.body_layout.setSpacing(4)
        self.body_container.setVisible(False)
        root.addWidget(self.body_container)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.body_container.setVisible(expanded)
        self.expand_btn.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.expand_btn.setToolTip("Collapse" if expanded else "Expand")
        self._description_label.setVisible(bool(self._description_label.text()) and not expanded)

    def set_description(self, text: str) -> None:
        self._description_label.setText(text)
        self._description_label.setVisible(bool(text) and not self._expanded)

    def set_body_widget(self, body: QWidget) -> None:
        self.body_layout.addWidget(body)


class RemovableCardWidget(BaseCardWidget):
    """A card with an "X" in its header that asks to be removed."""

    remove_requested = pyqtSignal()

    def __init__(self, title: str, parent: QWidget) -> None:
        super().__init__(title, parent)
        remove_btn = QToolButton(self)
        remove_btn.setAutoRaise(True)
        remove_btn.setText("X")
        remove_btn.setStyleSheet(f"QToolButton {{ color: {_REMOVE_BTN_COLOR}; font-weight: 700; }}")
        remove_btn.setToolTip("Remove")
        remove_btn.clicked.connect(self.remove_requested)
        self.header_row.addWidget(remove_btn)
