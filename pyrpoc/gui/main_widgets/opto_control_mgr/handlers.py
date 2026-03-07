from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox

from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.gui.main_widgets.opto_control_mgr.instance_card import InstanceCardWidget

if TYPE_CHECKING:
    from pyrpoc.gui.main_widgets.opto_control_mgr.widget import OptoControlManagerWidget


def refresh_available(widget: OptoControlManagerWidget) -> None:
    '''Read selected modality compatibility and refresh dropdown entries only for those.

    This is called by the manager on startup and whenever modality changes.
    '''
    contract = widget.modality_service.get_selected_contract()
    allowed = contract.get("allowed_optocontrols", [])
    allowed_keys = {
        getattr(opto_cls, "OPTOCONTROL_KEY", "")
        for opto_cls in allowed
        if hasattr(opto_cls, "OPTOCONTROL_KEY")
    }

    current_key = widget._selected_type_key()
    widget.type_combo.clear()
    for row in widget.opto_control_service.list_available():
        key = row["key"]
        if allowed_keys and key not in allowed_keys:
            continue
        name = row.get("display_name", key)
        widget.type_combo.addItem(name, key)

    if current_key:
        idx = widget.type_combo.findData(current_key)
        if idx >= 0:
            widget.type_combo.setCurrentIndex(idx)


def refresh_instances(widget: OptoControlManagerWidget) -> None:
    '''
    Keep card widgets stable across inventory deltas.

    Route:
    - service emits `inventory_changed` after add/remove
    - this function compares service rows with existing cards and only adds/removes
      changed control instances, preserving existing expanded bodies and QObject state.
    '''
    desired_cards: dict[BaseOptoControl, InstanceCardWidget] = {}
    rows = widget.opto_control_service.list_instances()
    wanted_controls = [row["state"] for row in rows]

    _remove_missing_cards(widget, set(wanted_controls))

    for row in rows:
        control: BaseOptoControl = row["state"]
        title = row["name"]
        card = widget.state.card_widgets.get(control)
        if card is None:
            card = _create_card(widget, control, title, row["key"])
            card.set_enable_checked(row.get("enabled", False))
            card.set_enable_visible(True)
            card.set_marker_text(f"[{row['key']}]")
            widget.state.card_widgets[control] = card

        card.set_enable_checked(row.get("enabled", False))
        _refresh_card_text(card, title, row["key"])
        desired_cards[control] = card

    _reorder_cards(widget, desired_cards, rows)


def on_expand_requested(widget: OptoControlManagerWidget, state_obj: object) -> None:
    '''Toggle the card body and lazy-build the control widget when expanding.

    The body is created here so collapsed cards remain lightweight.
    '''
    if not isinstance(state_obj, BaseOptoControl):
        return
    card = widget.state.card_widgets.get(state_obj)
    if card is None:
        return

    card.set_expanded(not card.is_expanded())
    if not card.is_expanded():
        return

    if card.body_layout.count() != 0:
        if _is_stale_widget(card):
            card.set_body_widget(None)
            _reset_control_widget_cache(state_obj)
        else:
            card.set_local_status("Status: ready")
            return

    if _attach_widget_to_card(widget, state_obj, card):
        return

    # Retry path for stale-object style failures in case the cached control widget
    # wrapper was deleted after a prior layout cleanup.
    _reset_control_widget_cache(state_obj)
    _attach_widget_to_card(widget, state_obj, card)


def on_add_clicked(widget: OptoControlManagerWidget) -> None:
    '''Add the selected type into app state and trigger a list refresh.

    Route: Add button -> this handler -> service.create_opto_control -> inventory_changed signal.
    '''
    key = widget._selected_type_key()
    if not key:
        show_error(widget, "No opto-control available for current modality")
        return
    try:
        widget.opto_control_service.create_opto_control(key)
        widget.status_label.setText("Status: added opto-control")
    except Exception as exc:
        show_error(widget, str(exc))


def on_enable_toggled(widget: OptoControlManagerWidget, state_obj: object, checked: bool) -> None:
    '''Persist checkbox state directly into the control instance.

    Route: card checkbox -> this handler -> service.set_enabled.
    '''
    if not isinstance(state_obj, BaseOptoControl):
        return
    widget.opto_control_service.set_enabled(state_obj, checked)
    card = widget.state.card_widgets.get(state_obj)
    if card is not None:
        card.set_local_status(f"Status: {'enabled' if checked else 'disabled'}")


def on_remove_requested(widget: OptoControlManagerWidget, state_obj: object) -> None:
    '''Remove control instance from state and UI.

    Route: card remove button -> this handler -> service.remove_opto_control.
    '''
    if not isinstance(state_obj, BaseOptoControl):
        return
    widget.opto_control_service.remove_opto_control(state_obj)
    widget.status_label.setText("Status: removed opto-control")


def on_modality_selected(widget: OptoControlManagerWidget, key: str) -> None:
    del key
    refresh_available(widget)


def show_error(widget: OptoControlManagerWidget, message: str) -> None:
    widget.status_label.setText(f"Status: error - {message}")
    QMessageBox.critical(widget, "Opto-Control Error", message)


def _remove_missing_cards(widget: OptoControlManagerWidget, wanted_controls: set[BaseOptoControl]) -> None:
    '''
    Remove only cards whose control state is no longer present.

    This path is triggered from inventory refresh after service remove/add.
    Reusable cards are retained so expanded bodies are not force-destroyed.
    '''
    for control, card in list(widget.state.card_widgets.items()):
        if control not in wanted_controls:
            widget.state.card_widgets.pop(control)
            card.setParent(None)
            card.deleteLater()


def _create_card(
    widget: OptoControlManagerWidget,
    control: BaseOptoControl,
    title: str,
    key: str,
) -> InstanceCardWidget:
    '''
    Create one opto-control card for a newly-added control state.

    From:
    - add/remove refresh cycle (from service signal) -> this branch.
    To:
    - card instance owned by this manager so existing references map state->widget.
    '''
    card = InstanceCardWidget(control, title, widget)
    card.set_enable_visible(True)
    card.set_marker_text(f"[{key}]")
    card.remove_requested.connect(lambda state_obj, w=widget: on_remove_requested(w, state_obj))
    card.expand_requested.connect(lambda state_obj, w=widget: on_expand_requested(w, state_obj))
    card.enable_toggled.connect(
        lambda state_obj, checked, w=widget: on_enable_toggled(w, state_obj, checked)
    )
    widget.instances_layout.insertWidget(widget.instances_layout.count() - 1, card)
    return card


def _refresh_card_text(card: InstanceCardWidget, title: str, key: str) -> None:
    '''Keep card title/marker in sync with row metadata from service rows.'''
    card.title_label.setText(title)
    card.set_marker_text(f"[{key}]")


def _attach_widget_to_card(
    widget: OptoControlManagerWidget,
    state_obj: BaseOptoControl,
    card: InstanceCardWidget,
) -> bool:
    '''
    Resolve and mount a concrete control widget for an expanded card.

    Route:
    - UI expand event -> this method -> OptoControlService.get_widget
    - concrete control widget returned and inserted into existing card body layout.
    '''
    try:
        child_widget = widget.opto_control_service.get_widget(
            state_obj,
            parent=card.body_container,
        )
        card.set_body_widget(child_widget)
        card.set_local_status("Status: ready")
        return True
    except Exception as exc:
        card.set_local_status(f"Status: error - {exc}")
        return False


def _is_stale_widget(card: InstanceCardWidget) -> bool:
    '''Probe cached QWidget before trusting it after incremental updates.'''
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
    except Exception as exc:
        card.set_local_status(f"Status: warning - stale body widget detected ({exc})")
        return True
    return False


def _reset_control_widget_cache(state_obj: BaseOptoControl) -> None:
    '''
    Clear any cached widget reference on a control instance.

    This lets a stale deleted QWidget wrapper be rebuilt once when retrying attach.
    '''
    if hasattr(state_obj, "widget"):
        try:
            setattr(state_obj, "widget", None)
        except Exception:
            pass


def _reorder_cards(
    widget: OptoControlManagerWidget,
    desired_cards: dict[BaseOptoControl, InstanceCardWidget],
    rows: list[dict[str, object]],
) -> None:
    '''
    Preserve service ordering while reusing existing card objects.

    We intentionally detach cards from layout and insert again to change order
    without destroying card instances that may hold expanded child widgets.
    '''
    desired_order = [row["state"] for row in rows]

    while widget.instances_layout.count() > 1:
        item = widget.instances_layout.takeAt(0)
        card = item.widget() #pyright: ignore
        if card is not None:
            card.setParent(None)

    for state in desired_order:
        card = desired_cards.get(state)
        if card is not None:
            widget.instances_layout.insertWidget(widget.instances_layout.count(), card)

    # Keep the trailing stretch anchor after all cards.
    widget.instances_layout.addStretch(1)
