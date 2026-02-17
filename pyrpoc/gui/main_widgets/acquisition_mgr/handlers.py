from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox

from pyrpoc.gui.main_widgets.acquisition_mgr.forms import build_param_form, collect_values

if TYPE_CHECKING:
    from pyrpoc.gui.main_widgets.acquisition_mgr.widget import AcquisitionManagerWidget


def populate_modalities(widget: AcquisitionManagerWidget) -> None:
    current = widget.modality_combo.currentText()
    widget.modality_combo.blockSignals(True)
    widget.modality_combo.clear()
    rows = widget.modality_service.list_available()
    for row in rows:
        widget.modality_combo.addItem(row["key"])
    widget.modality_combo.blockSignals(False)

    if current and widget.modality_combo.findText(current) >= 0:
        widget.modality_combo.setCurrentText(current)
    elif widget.modality_combo.count() > 0:
        widget.modality_combo.setCurrentIndex(0)
        on_modality_selected(widget, widget.modality_combo.currentText())


def on_modality_selected(widget: AcquisitionManagerWidget, key: str) -> None:
    if not key:
        return
    try:
        widget.modality_service.select_modality(key)
    except Exception as exc:
        handle_error(widget, str(exc))


def handle_modality_selected(widget: AcquisitionManagerWidget, key: str) -> None:
    del key
    parameter_groups = widget.modality_service.get_selected_parameters()
    build_param_form(widget.ui, widget.state, parameter_groups)


def configure_modality(widget: AcquisitionManagerWidget) -> None:
    params = collect_values(widget.state.param_widgets)
    widget.modality_service.configure(params)


def on_start_clicked(widget: AcquisitionManagerWidget) -> None:
    try:
        configure_modality(widget)
        widget.modality_service.start()
        acquire_single_frame(widget)
        widget.modality_service.stop()
    except Exception as exc:
        handle_error(widget, str(exc))


def on_continuous_clicked(widget: AcquisitionManagerWidget) -> None:
    try:
        configure_modality(widget)
        widget.modality_service.start()
        timer = widget.state.continuous_timer
        if timer is not None:
            timer.start()
        widget.status_label.setText("Status: continuous acquisition")
    except Exception as exc:
        handle_error(widget, str(exc))


def acquire_single_frame(widget: AcquisitionManagerWidget) -> None:
    try:
        widget.modality_service.acquire_once()
    except Exception as exc:
        timer = widget.state.continuous_timer
        if timer is not None:
            timer.stop()
        try:
            widget.modality_service.stop()
        except Exception as stop_exc:
            widget.status_label.setText(f"Status: stop error - {stop_exc}")
        handle_error(widget, str(exc))


def on_stop_clicked(widget: AcquisitionManagerWidget) -> None:
    timer = widget.state.continuous_timer
    if timer is not None:
        timer.stop()
    try:
        widget.modality_service.stop()
    except Exception as exc:
        handle_error(widget, str(exc))


def handle_requirements_changed(
    widget: AcquisitionManagerWidget,
    ok: bool,
    missing_names: list[str],
) -> None:
    if ok:
        widget.status_label.setText("Status: ready")
        return
    text = ", ".join(missing_names)
    widget.status_label.setText(f"Status: missing instruments -> {text}")


def handle_error(widget: AcquisitionManagerWidget, message: str) -> None:
    widget.status_label.setText(f"Status: error - {message}")
    QMessageBox.critical(widget, "Acquisition Error", message)


def on_service_error(widget: AcquisitionManagerWidget, message: str) -> None:
    widget.status_label.setText(f"Status: error - {message}")
