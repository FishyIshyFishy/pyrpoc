from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtWidgets import QLabel, QMessageBox

from pyrpoc.displays.display_registry import display_registry
from pyrpoc.displays.base_display import BaseDisplay
from pyrpoc.gui.main_widgets.instance_card import RemovableCardWidget as DisplayCardWidget
from pyrpoc.gui.main_widgets.display_mgr.forms import prompt_display_parameters

if TYPE_CHECKING:
    from pyrpoc.gui.main_widgets.display_mgr.widget import DisplayManagerWidget


# ---------------------------------------------------------------------------
# Refresh helpers
# ---------------------------------------------------------------------------

def refresh_available(widget: DisplayManagerWidget) -> None:
    """
    Build display dropdown constrained by selected modality contract.
    """
    contract = widget.modality_service.get_selected_contract()
    emitted_kinds = contract.get("emitted_kinds", [])
    allowed_displays = set(contract.get("allowed_displays", []))
    current_key = widget._selected_display_key()
    available_rows = {row["key"]: row for row in widget.display_service.list_available()}

    widget.display_combo.clear()
    if emitted_kinds:
        keys = widget.display_service.list_compatible_with(emitted_kinds)
    else:
        keys = list(available_rows.keys())

    if allowed_displays:
        keys = [key for key in keys if key in allowed_displays]

    for key in keys:
        row = available_rows.get(key, {"display_name": key})
        widget.display_combo.addItem(row.get("display_name", key), key)

    if current_key:
        idx = widget.display_combo.findData(current_key)
        if idx >= 0:
            widget.display_combo.setCurrentIndex(idx)


def refresh_instances(widget: DisplayManagerWidget) -> None:
    """
    Diff-update card list from service inventory.  Existing cards are reused
    so expanded bodies survive add/remove cycles.
    """
    rows = widget.display_service.list_instances()
    wanted_displays: set[object] = set()

    for row in rows:
        state = _row_state(row)
        if state is not None:
            wanted_displays.add(state)

    _remove_missing_cards(widget, wanted_displays)

    desired_cards: dict[object, DisplayCardWidget] = {}
    for row in rows:
        state = _row_state(row)
        if state is None:
            continue

        title = row.get("name", "Display")
        attached = bool(row.get("attached", False))

        card = widget.state.card_widgets.get(state)
        if card is None:
            card = _create_card(widget, state, str(title))
            widget.state.card_widgets[state] = card

        card.title_label.setText(str(title))
        card.set_toggle_checked(attached)
        desired_cards[state] = card

    _reorder_cards(widget, desired_cards, rows)


# ---------------------------------------------------------------------------
# Button/action handlers
# ---------------------------------------------------------------------------

def on_add_clicked(widget: DisplayManagerWidget) -> None:
    key = widget._selected_display_key()
    if not key:
        show_error(widget, "No compatible display available for selected modality")
        return

    raw_settings: dict[str, Any] = {}
    display_cls = display_registry.get_class(key)
    row = widget.display_service.list_available()
    fallback = next((r for r in row if r["key"] == key), None)
    display_name = str(fallback.get("display_name", key)) if isinstance(fallback, dict) else key

    if any(display_cls.DISPLAY_PARAMETERS.values()):
        settings = prompt_display_parameters(widget, display_cls.DISPLAY_PARAMETERS)
        if settings is None:
            return
        raw_settings = settings

    try:
        widget.display_service.create_display(key, raw_settings, user_label=display_name)
    except Exception as exc:
        show_error(widget, str(exc))


def on_attach_toggled(widget: DisplayManagerWidget, state_obj: object, checked: bool) -> None:
    if not isinstance(state_obj, BaseDisplay):
        return
    if checked:
        widget.display_service.attach(state_obj)
    else:
        widget.display_service.detach(state_obj)


def on_remove_requested(widget: DisplayManagerWidget, state_obj: object) -> None:
    if not isinstance(state_obj, BaseDisplay):
        return
    widget.display_service.remove_display(state_obj)


def on_expand_requested(widget: DisplayManagerWidget, state_obj: object) -> None:
    card = widget.state.card_widgets.get(state_obj)
    if card is None:
        return
    card.set_expanded(not card.is_expanded())
    if not card.is_expanded():
        return
    if card.body_layout.count() == 0:
        _build_placeholder_body(card)


def on_display_error(widget: DisplayManagerWidget, _state: object, message: str) -> None:
    show_error(widget, message)


def on_modality_selected(widget: DisplayManagerWidget, key: str) -> None:
    del key
    refresh_available(widget)


def show_error(widget: DisplayManagerWidget, message: str) -> None:
    QMessageBox.critical(widget, "Display Error", message)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _row_state(row: dict[str, Any]) -> object | None:
    """Extract a stable identity object from a service row."""
    state = row.get("state")
    if state is not None:
        return state
    display_id = row.get("display_id")
    return display_id if display_id is not None else None


def _remove_missing_cards(widget: DisplayManagerWidget, wanted: set[object]) -> None:
    for state, card in list(widget.state.card_widgets.items()):
        if state not in wanted:
            widget.state.card_widgets.pop(state)
            card.setParent(None)  # type: ignore[arg-type]
            card.deleteLater()


def _create_card(
    widget: DisplayManagerWidget,
    state_obj: object,
    title: str,
) -> DisplayCardWidget:
    card = DisplayCardWidget(state_obj, title, widget)
    card.toggle_changed.connect(
        lambda obj, checked, w=widget: on_attach_toggled(w, obj, checked)
    )
    card.remove_requested.connect(
        lambda obj, w=widget: on_remove_requested(w, obj)
    )
    card.expand_requested.connect(
        lambda obj, w=widget: on_expand_requested(w, obj)
    )
    widget.instances_layout.insertWidget(widget.instances_layout.count(), card)
    return card


def _build_placeholder_body(card: DisplayCardWidget) -> None:
    """Insert a placeholder settings panel into an expanded card body."""
    placeholder = QLabel("Display settings — coming soon", card.body_container)
    placeholder.setStyleSheet("color: palette(mid); font-style: italic; padding: 4px 0;")
    card.set_body_widget(placeholder)


def _reorder_cards(
    widget: DisplayManagerWidget,
    desired_cards: dict[object, DisplayCardWidget],
    rows: list[dict[str, Any]],
) -> None:
    desired_order = [_row_state(row) for row in rows]

    while widget.instances_layout.count() > 0:
        item = widget.instances_layout.takeAt(0)
        card = item.widget()  # type: ignore[assignment]
        if card is not None:
            card.setParent(None)  # type: ignore[arg-type]

    for state in desired_order:
        card = desired_cards.get(state)  # type: ignore[arg-type]
        if card is not None:
            widget.instances_layout.insertWidget(widget.instances_layout.count(), card)
