"""Instrument-manager logic, rewired onto the new backend (Option B).

The widget and layout are the original's; only the backend calls change:
``InstrumentService`` -> ``instrument_registry`` + ``AppState.instruments``. Add/remove
mutate app state directly and refresh, since there is no external inventory signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QWidget

from pyrpoc_next.gui.widgets.cards import RemovableCardWidget
from pyrpoc_next.instruments import instrument_registry
from pyrpoc_next.instruments.base import Instrument
from pyrpoc_next.structs.keys import InstrumentKey

if TYPE_CHECKING:
    from pyrpoc_next.gui.panels.instruments.widget import InstrumentManagerWidget


def class_for(key: InstrumentKey) -> type[Instrument]:
    return instrument_registry.entries[key]


def refresh_available(widget: InstrumentManagerWidget) -> None:
    """Populate the dropdown from the registry, preserving the current selection."""
    current = widget.selected_key()
    widget.type_combo.blockSignals(True)
    widget.type_combo.clear()
    for key in instrument_registry.available():
        widget.type_combo.addItem(class_for(key).display_name, key)
    widget.type_combo.blockSignals(False)

    if current is not None:
        idx = widget.type_combo.findData(current)
        if idx >= 0:
            widget.type_combo.setCurrentIndex(idx)
    elif widget.type_combo.count() > 0:
        widget.type_combo.setCurrentIndex(0)


def refresh_instances(widget: InstrumentManagerWidget) -> None:
    """Sync cards with ``app_state.instruments``, reusing cards for unchanged instances."""
    instruments = list(widget.app_state.instruments)
    remove_missing_cards(widget, set(instruments))

    for instrument in instruments:
        card = widget.state.card_widgets.get(instrument)
        if card is None:
            card = create_card(widget, instrument)
            widget.state.card_widgets[instrument] = card
        refresh_card_text(card, instrument)

    reorder_cards(widget, instruments)


def on_add_clicked(widget: InstrumentManagerWidget) -> None:
    """Create the selected instrument, register it in app state, refresh cards."""
    key = widget.selected_key()
    if key is None:
        return
    try:
        instrument = instrument_registry.create(key)
    except Exception as exc:
        QMessageBox.critical(widget, "Instrument Error", str(exc))
        return
    widget.app_state.instruments.append(instrument)
    refresh_instances(widget)


def on_remove_requested(widget: InstrumentManagerWidget, instrument: object) -> None:
    """Remove an instrument from app state and refresh cards."""
    if instrument in widget.app_state.instruments:
        widget.app_state.instruments.remove(instrument)
    refresh_instances(widget)


def on_expand_requested(widget: InstrumentManagerWidget, instrument: object) -> None:
    """Toggle a card; build its body (summary + Test Connection) on first expand."""
    card = widget.state.card_widgets.get(instrument)
    if card is None or not isinstance(instrument, Instrument):
        return
    card.set_expanded(not card.is_expanded())
    if card.is_expanded() and card.body_layout.count() == 0:
        card.set_body_widget(build_body(instrument, card))


def build_body(instrument: Instrument, card: RemovableCardWidget) -> QWidget:
    """The minimal expanded body: a status line and a Test Connection button."""
    body = QWidget()
    layout = QHBoxLayout(body)
    layout.setContentsMargins(0, 0, 0, 0)
    status = QLabel(instrument.summary())
    test = QPushButton("Test Connection")

    def run_test() -> None:
        instrument.test_connection()
        status.setText(instrument.summary())
        refresh_card_text(card, instrument)

    test.clicked.connect(run_test)
    layout.addWidget(status, 1)
    layout.addWidget(test)
    return body


def create_card(widget: InstrumentManagerWidget, instrument: Instrument) -> RemovableCardWidget:
    card = RemovableCardWidget(instrument, instrument.display_name, widget)
    card.set_toggle_visible(False)
    card.remove_requested.connect(lambda obj, w=widget: on_remove_requested(w, obj))
    card.expand_requested.connect(lambda obj, w=widget: on_expand_requested(w, obj))
    refresh_card_text(card, instrument)
    return card


def refresh_card_text(card: RemovableCardWidget, instrument: Instrument) -> None:
    card.title_label.setText(instrument.display_name)
    card.set_local_status(instrument.summary())


def remove_missing_cards(widget: InstrumentManagerWidget, wanted: set) -> None:
    for instrument, card in list(widget.state.card_widgets.items()):
        if instrument not in wanted:
            widget.state.card_widgets.pop(instrument)
            card.setParent(None)
            card.deleteLater()


def reorder_cards(widget: InstrumentManagerWidget, order: list) -> None:
    """Rebuild layout order to match app state without recreating card instances."""
    while (item := widget.instances_layout.takeAt(0)) is not None:
        item.widget()  # detach without deleting; cards live in the card map
    for instrument in order:
        card = cast("RemovableCardWidget | None", widget.state.card_widgets.get(instrument))
        if card is not None:
            widget.instances_layout.addWidget(card)
