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
    '''Rebuild control cards from `app_state.optocontrols`.

    Called after add/remove, and each time the service emits `inventory_changed`.
    '''
    _clear_instance_cards(widget)
    for row in widget.opto_control_service.list_instances():
        control: BaseOptoControl = row["state"]
        title = row["name"]
        card = InstanceCardWidget(control, title, widget)
        card.set_enable_checked(row.get("enabled", False))
        card.set_enable_visible(True)
        card.set_marker_text(f"[{row['key']}]")
        card.remove_requested.connect(lambda state_obj, w=widget: on_remove_requested(w, state_obj))
        card.expand_requested.connect(
            lambda state_obj, w=widget: on_expand_requested(w, state_obj)
        )
        card.enable_toggled.connect(
            lambda state_obj, checked, w=widget: on_enable_toggled(w, state_obj, checked)
        )
        widget.state.card_widgets[control] = card
        widget.instances_layout.insertWidget(widget.instances_layout.count() - 1, card)


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

    if card.body_layout.count() == 0:
        try:
            child_widget = widget.opto_control_service.get_widget(
                state_obj,
                parent=card.body_container,
            )
            card.set_body_widget(child_widget)
            card.set_local_status("Status: ready")
        except Exception as exc:
            card.set_local_status(f"Status: error - {exc}")


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


def _clear_instance_cards(widget: OptoControlManagerWidget) -> None:
    widget.state.card_widgets.clear()
    while widget.instances_layout.count() > 1:
        item = widget.instances_layout.takeAt(0)
        card = item.widget() #pyright: ignore
        if card is not None:
            card.setParent(None)
            card.deleteLater()
