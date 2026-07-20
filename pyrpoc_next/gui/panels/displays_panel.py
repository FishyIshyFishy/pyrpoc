"""Displays tab: add displays (each opens as its own dock) and remove them."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc_next.gui.displays import display_registry
from pyrpoc_next.gui.widgets.cards import RemovableCardWidget


class DisplaysPanel(QWidget):
    """Add displays as docks; list them as removable cards."""

    def __init__(self, state, open_dock: Callable[[QWidget, str], object], close_dock: Callable[[object], None]):
        super().__init__()
        self.state = state
        self.open_dock = open_dock
        self.close_dock = close_dock

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.combo = QComboBox()
        for key in display_registry.available():
            self.combo.addItem(key.value, key)
        add = QPushButton("Add")
        add.clicked.connect(self.add)
        top.addWidget(self.combo, 1)
        top.addWidget(add)
        root.addLayout(top)

        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.cards_container)
        root.addWidget(scroll, 1)

    def add(self) -> None:
        display = display_registry.create(self.combo.currentData())
        self.state.displays.append(display)
        dock = self.open_dock(display, display.manifest.display_name)

        card = RemovableCardWidget(display, display.manifest.display_name)
        card.set_toggle_visible(False)
        card.expand_btn.setVisible(False)
        card.remove_requested.connect(lambda _: self.remove(display, dock, card))
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def remove(self, display, dock, card) -> None:
        if display in self.state.displays:
            self.state.displays.remove(display)
        self.close_dock(dock)
        card.setParent(None)
