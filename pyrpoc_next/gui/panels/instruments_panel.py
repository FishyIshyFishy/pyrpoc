"""Instruments tab: add instruments as removable cards, test their connection."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc_next.gui.widgets.cards import RemovableCardWidget
from pyrpoc_next.instruments import instrument_registry


class InstrumentsPanel(QWidget):
    """Add/remove instruments and test connections, using the shared card look."""

    def __init__(self, state):
        super().__init__()
        self.state = state
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.combo = QComboBox()
        for key in instrument_registry.available():
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
        instrument = instrument_registry.create(self.combo.currentData())
        self.state.instruments.append(instrument)
        self.add_card(instrument)

    def add_card(self, instrument) -> None:
        card = RemovableCardWidget(instrument, instrument.display_name)
        card.set_toggle_visible(False)
        card.expand_requested.connect(lambda _: card.set_expanded(not card.is_expanded()))
        card.remove_requested.connect(lambda _: self.remove(instrument, card))

        body = QWidget()
        layout = QHBoxLayout(body)
        status = QLabel(instrument.summary())
        test = QPushButton("Test Connection")
        test.clicked.connect(lambda: (instrument.test_connection(), status.setText(instrument.summary())))
        layout.addWidget(status, 1)
        layout.addWidget(test)
        card.set_body_widget(body)

        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def remove(self, instrument, card) -> None:
        if instrument in self.state.instruments:
            self.state.instruments.remove(instrument)
        card.setParent(None)
