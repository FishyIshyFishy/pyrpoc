"""Display-manager logic, rewired onto the new backend (Option B).

The card list + attach toggle are the original's; only the backend calls change:
DisplayService -> display_registry + AppState.displays, and each display opens/closes
as a dock via the callbacks the window hands the widget. The attach toggle flips
``display.attached``, which gates live rendering in the display base.

Modality-compatibility filtering of the dropdown is deferred to Phase 4 (it needs the
routine/block UI); until then all displays are offered and Controller.check() catches
mismatches at play time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyrpoc_next.gui.displays import display_registry
from pyrpoc_next.gui.displays.base import DisplayWidget
from pyrpoc_next.gui.widgets.cards import RemovableCardWidget

if TYPE_CHECKING:
    from pyrpoc_next.gui.panels.displays.widget import DisplayManagerWidget


def refresh_available(widget: DisplayManagerWidget) -> None:
    """Populate the dropdown from the registry, preserving the current selection."""
    current = widget.selected_key()
    widget.display_combo.blockSignals(True)
    widget.display_combo.clear()
    for key in display_registry.available():
        widget.display_combo.addItem(display_registry.entries[key].manifest.display_name, key)
    widget.display_combo.blockSignals(False)
    if current is not None:
        idx = widget.display_combo.findData(current)
        if idx >= 0:
            widget.display_combo.setCurrentIndex(idx)


def refresh_instances(widget: DisplayManagerWidget) -> None:
    """Sync cards with ``app_state.displays``, reusing cards for unchanged displays."""
    displays = list(widget.app_state.displays)
    remove_missing_cards(widget, set(displays))

    for display in displays:
        card = widget.state.card_widgets.get(display)
        if card is None:
            card = create_card(widget, display)
            widget.state.card_widgets[display] = card
        card.title_label.setText(display.manifest.display_name)
        card.set_toggle_checked(bool(display.attached))

    reorder_cards(widget, displays)


def on_add_clicked(widget: DisplayManagerWidget) -> None:
    """Create the selected display, open its dock, register it in app state."""
    key = widget.selected_key()
    if key is None:
        return
    display = display_registry.create(key)
    widget.app_state.displays.append(display)
    widget.docks[display] = widget.open_dock(display, display.manifest.display_name)
    refresh_instances(widget)


def on_attach_toggled(widget: DisplayManagerWidget, display: object, checked: bool) -> None:
    """The attach toggle gates whether the display receives live parcels."""
    if isinstance(display, DisplayWidget):
        display.attached = bool(checked)


def on_remove_requested(widget: DisplayManagerWidget, display: object) -> None:
    """Remove a display from app state and close its dock."""
    if display in widget.app_state.displays:
        widget.app_state.displays.remove(display)
    dock = widget.docks.pop(display, None)
    if dock is not None:
        widget.close_dock(dock)
    refresh_instances(widget)


def create_card(widget: DisplayManagerWidget, display: DisplayWidget) -> RemovableCardWidget:
    card = RemovableCardWidget(display, display.manifest.display_name, widget)
    card.expand_btn.setVisible(False)  # display controls live in the dock, not the card
    card.set_toggle_visible(True)  # the toggle is the attach control
    card.set_toggle_checked(bool(display.attached))
    card.toggle_changed.connect(lambda obj, checked, w=widget: on_attach_toggled(w, obj, checked))
    card.remove_requested.connect(lambda obj, w=widget: on_remove_requested(w, obj))
    return card


def remove_missing_cards(widget: DisplayManagerWidget, wanted: set) -> None:
    for display, card in list(widget.state.card_widgets.items()):
        if display not in wanted:
            widget.state.card_widgets.pop(display)
            card.setParent(None)
            card.deleteLater()


def reorder_cards(widget: DisplayManagerWidget, order: list) -> None:
    while (item := widget.instances_layout.takeAt(0)) is not None:
        item.widget()  # detach without deleting; cards live in the card map
    for display in order:
        card = widget.state.card_widgets.get(display)
        if card is not None:
            widget.instances_layout.addWidget(card)
