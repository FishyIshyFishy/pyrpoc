"""Collapsible cards: an expand arrow and a title, a summary shown while collapsed,
and a body shown while expanded. The whole header toggles the card."""

from __future__ import annotations

from html import escape

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPalette
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

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


class CardHeader(QWidget):
    """A card's title row; a click anywhere on it that no button took asks to toggle."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        super().mousePressEvent(event)
        self.clicked.emit()


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

        self.header = CardHeader(self)
        self.header.clicked.connect(lambda: self.set_expanded(not self._expanded))
        self.header_row = QHBoxLayout(self.header)
        self.header_row.setContentsMargins(0, 0, 0, 0)
        self.header_row.setSpacing(4)
        self.expand_btn = QToolButton(self)
        self.expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.expand_btn.setAutoRaise(True)
        self.expand_btn.setToolTip("Expand")
        self.expand_btn.clicked.connect(lambda: self.set_expanded(not self._expanded))
        self.header_row.addWidget(self.expand_btn)
        self.title_label = QLabel(title, self)
        title_font = self.title_label.font()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.header_row.addWidget(self.title_label, 1)
        root.addWidget(self.header)

        self._summary: list[tuple[str, str]] = []
        self._description_label = QLabel("", self)
        self._description_label.setStyleSheet("font-size: 9pt; padding-left: 22px;")
        self._description_label.setWordWrap(True)
        # Wrapped text would otherwise hold the card wider than a narrow panel.
        self._description_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
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
        self._description_label.setVisible(bool(self._summary) and not expanded)

    def set_summary(self, pairs: list[tuple[str, str]]) -> None:
        """Show ``(name, value)`` pairs under the title while collapsed: names muted, values
        plain, so the current settings can be read at a glance."""
        self._summary = pairs
        self.render_summary()
        self._description_label.setVisible(bool(pairs) and not self._expanded)

    def render_summary(self) -> None:
        # Rich text cannot use ``palette()``, so the muted colour is read here and re-read when
        # the theme changes.
        muted = self.palette().color(QPalette.ColorRole.Mid).name()
        parts = [
            f'<span style="color:{muted}">{escape(name).replace(" ", "&nbsp;")}</span>'
            f"&nbsp;{escape(value).replace(' ', '&nbsp;')}"
            for name, value in self._summary
        ]
        self._description_label.setText("&nbsp;· ".join(parts))

    def changeEvent(self, event: QEvent | None) -> None:
        super().changeEvent(event)
        if event is not None and event.type() == QEvent.Type.PaletteChange:
            self.render_summary()

    def set_body_widget(self, body: QWidget) -> None:
        self.body_layout.addWidget(body)


class RemovableCardWidget(BaseCardWidget):
    """A card with an "X" in its header that asks to be removed."""

    remove_requested = pyqtSignal()

    def __init__(self, title: str, parent: QWidget) -> None:
        super().__init__(title, parent)
        remove_btn = QToolButton(self)
        remove_btn.setAutoRaise(True)
        remove_btn.setText("✕")
        remove_btn.setStyleSheet(f"QToolButton {{ color: {_REMOVE_BTN_COLOR}; font-weight: 700; }}")
        remove_btn.setToolTip("Remove")
        remove_btn.clicked.connect(self.remove_requested)
        self.header_row.addWidget(remove_btn)
