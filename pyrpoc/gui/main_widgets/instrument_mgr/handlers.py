from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

from PyQt6.QtWidgets import QMessageBox

from pyrpoc.domain.app_state import InstrumentState
from pyrpoc.gui.main_widgets.opto_control_mgr.instance_card import InstanceCardWidget

if TYPE_CHECKING:
    from pyrpoc.gui.main_widgets.instrument_mgr.widget import InstrumentManagerWidget


def refresh_available(widget: InstrumentManagerWidget) -> None:
    """Populate dropdown from registered instrument classes.

    Route:
    - widget startup / state refresh -> this function -> `InstrumentService.list_available`
    - contract comes from registry metadata, no direct instrument imports in manager.
    """
    current_key = widget._selected_type_key()
    widget.type_combo.blockSignals(True)
    widget.type_combo.clear()
    for row in widget.instrument_service.list_available():
        key = row["key"]
        name = row.get("display_name", key)
        widget.type_combo.addItem(name, key)
    widget.type_combo.blockSignals(False)

    if current_key:
        idx = widget.type_combo.findData(current_key)
        if idx >= 0:
            widget.type_combo.setCurrentIndex(idx)
    elif widget.type_combo.count() > 0:
        widget.type_combo.setCurrentIndex(0)


def refresh_instances(widget: InstrumentManagerWidget) -> None:
    """Synchronize card widgets with current service inventory.

    Route:
    - service emits `inventory_changed` after add/remove
    - manager diffs desired state against existing cards and only creates/removes
      changed rows, preserving expanded bodies for unchanged rows.
    """
    desired_cards: dict[InstrumentState, InstanceCardWidget] = {}
    rows = widget.instrument_service.list_instances()
    wanted_states = [row["state"] for row in rows]

    _remove_missing_cards(widget, set(wanted_states))

    for row in rows:
        state: InstrumentState = row["state"]
        name = row["name"]

        card = widget.state.card_widgets.get(state)
        if card is None:
            card = _create_card(widget, state, name, row["key"])
            widget.state.card_widgets[state] = card

        _refresh_card_text(card, name, row["key"])
        desired_cards[state] = card

    _reorder_cards(widget, desired_cards, rows)


def on_expand_requested(widget: InstrumentManagerWidget, state_obj: object) -> None:
    """Handle lazy widget creation for a card body.

    Route:
    - user presses Expand button on card -> this handler -> `InstrumentService.get_widget`
    - result widget is inserted into card body and rendered there.
    """
    if not isinstance(state_obj, InstrumentState):
        return
    card: InstanceCardWidget = widget.state.card_widgets.get(state_obj) # pyright: ignore
    if card is None:
        return

    card.set_expanded(not card.is_expanded())
    if not card.is_expanded():
        return

    if card.body_layout.count() != 0:
        if _is_stale_widget(card):
            card.set_body_widget(None)
            _reset_instrument_widget_cache(state_obj)
            _attach_widget_to_card(widget, state_obj, card)
            return
        card.set_local_status("Status: ready")
        return

    if _attach_widget_to_card(widget, state_obj, card):
        return

    # Retry path for stale PyQt objects: clear any stale body child, force widget
    # refresh on the underlying instance, and build again once.
    _reset_instrument_widget_cache(state_obj)
    _attach_widget_to_card(widget, state_obj, card)


def on_add_clicked(widget: InstrumentManagerWidget) -> None:
    """Add selected instrument class into app state.

    Route:
    - Add button click -> this handler -> `InstrumentService.create_instrument`
    - service emits inventory signal for card list refresh.
    """
    key = widget._selected_type_key()
    if not key:
        return
    try:
        widget.instrument_service.create_instrument(key)
        widget.status_label.setText("Status: added instrument")
    except Exception as exc:
        show_error(widget, str(exc))


def on_remove_requested(widget: InstrumentManagerWidget, state_obj: object) -> None:
    """Remove an instrument instance and persist via service.

    Route:
    - card Remove button -> this handler -> `InstrumentService.remove_instrument`
    """
    if isinstance(state_obj, InstrumentState):
        widget.instrument_service.remove_instrument(state_obj)
        widget.status_label.setText("Status: removed instrument")


def show_error(widget: InstrumentManagerWidget, message: str) -> None:
    """Render a modal error and reflect it in the manager status label."""
    widget.status_label.setText(f"Status: error - {message}")
    QMessageBox.critical(widget, "Instrument Error", message)


def _attach_widget_to_card(widget: InstrumentManagerWidget, state_obj: InstrumentState, card: InstanceCardWidget) -> bool:
    """Build one concrete widget body for an expanded card.

    Returns True if widget could be instantiated and attached.
    """
    try:
        child = widget.instrument_service.get_widget(
            state_obj,
            parent=card.body_container,
            on_change=lambda: widget.status_label.setText("Status: widget changed"),
        )
        card.set_body_widget(child)
        card.set_local_status("Status: ready")
        return True
    except Exception as exc:
        # If this fails with a deleted-object style exception, caller may clear
        # cached widget state on the instrument and retry once.
        card.set_local_status(f"Status: error - {exc}")
        return False


def _is_stale_widget(card: InstanceCardWidget) -> bool:
    """Probe body widget liveness before trusting cached wrapper references."""
    if card.body_layout.count() == 0:
        return False
    widget_item = card.body_layout.itemAt(0)
    if widget_item is None:
        return False
    body_widget = widget_item.widget()
    if body_widget is None:
        return False
    try:
        body_widget.isVisible()
    except Exception as exc:  # typically PyQt wrapper-after-delete errors
        _mark_stale_status(card, f"Status: warning - stale body widget detected ({exc})")
        return True
    return False


def _mark_stale_status(card: InstanceCardWidget, status: str) -> None:
    """Attach a diagnostic status to the card while keeping manager flow alive."""
    card.set_local_status(status)


def _reset_instrument_widget_cache(state_obj: InstrumentState) -> None:
    """Reset cached widget state on the concrete instrument object if present.

    This is used only for stale-object recovery, e.g., when an old QWidget wrapper
    was deleted after an aggressive card rebuild path.
    """
    if hasattr(state_obj.instance, "widget"):
        try:
            setattr(state_obj.instance, "widget", None)
        except Exception:
            pass


def _remove_missing_cards(widget: InstrumentManagerWidget, wanted_states: set[InstrumentState]) -> None:
    """Delete only cards for states that no longer exist in service inventory."""
    for state, card in list(widget.state.card_widgets.items()):
        if state not in wanted_states:
            widget.state.card_widgets.pop(state)
            card.setParent(None)
            card.deleteLater()


def _create_card(
    widget: InstrumentManagerWidget,
    state: InstrumentState,
    name: str,
    key: str,
) -> InstanceCardWidget:
    """Create a new card for a new instrument state."""
    card = InstanceCardWidget(state, name, widget)
    card.set_enable_visible(False)
    card.set_marker_text(f"[{key}]")
    card.remove_requested.connect(lambda state_obj, w=widget: on_remove_requested(w, state_obj))
    card.expand_requested.connect(lambda state_obj, w=widget: on_expand_requested(w, state_obj))
    return card


def _refresh_card_text(card: InstanceCardWidget, name: str, key: str) -> None:
    """Keep visible card metadata aligned with registry metadata."""
    card.title_label.setText(name)
    card.set_marker_text(f"[{key}]")


def _reorder_cards(
    widget: InstrumentManagerWidget,
    desired_cards: dict[InstrumentState, InstanceCardWidget],
    rows: list[dict[str, object]],
) -> None:
    """Rebuild layout order to match service rows without recreating card instances."""
    desired_order: list[InstrumentState] = [row["state"] for row in rows]

    # Lift widgets out of layout while preserving instances.
    current_cards: list[InstanceCardWidget] = []
    while widget.instances_layout.count() > 0:
        item = widget.instances_layout.takeAt(0)
        card = item.widget() # pyright: ignore
        if card is not None:
            current_cards.append(card)

    for state in desired_order:
        card = desired_cards.get(state)
        if card is None:
            continue
        widget.instances_layout.insertWidget(widget.instances_layout.count(), card)

    if widget.instances_layout.count() == 0:
        widget.instances_layout.addStretch(1)
    else:
        widget.instances_layout.addStretch(1)
