"""The acquisition panel: pick a program, name the run, set it up, run it.

The transport row draws whatever controls the program's runners asked for;
Stop is fixed, since every run can be stopped. The name and save switch sit
there rather than in the form because they are not parameters of any program.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
)

from pyrpoc.structs.panel import Panel
from pyrpoc.structs.params import FieldContext
from pyrpoc.structs.registries import program_registry
from pyrpoc.structs.runner import Control

from ..components.icons import asset_icon
from ..components.param_form import ParamForm

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.app.model.application import Application


class AcquisitionPanel(Panel):
    display_name = "Acquisition"

    def __init__(self, app: Application):
        super().__init__()
        self.app = app
        self.form: ParamForm | None = None
        # Each rendered control, its button, and the watcher keeping them in line.
        self.control_buttons: list[tuple[Control, QPushButton, Callable[[], None]]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addLayout(self.build_program_row())
        root.addLayout(self.build_transport_row())
        self.status_label = QLabel("Status: idle", self)
        root.addWidget(self.status_label)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        root.addWidget(self.scroll_area, 1)

        self.connect_signals()
        self.render_controls()
        self.set_running_ui(False)
        self.on_save_changed()
        if self.app.selected_program is None:
            self.app.select_program(program_registry.keys()[0])
        else:
            self.on_program_selected(self.app.selected_program)

    def build_program_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Program:", self))
        self.program_combo = QComboBox(self)
        for key in program_registry.keys():
            self.program_combo.addItem(program_registry.get(key).display_name, key)
        row.addWidget(self.program_combo, 1)
        return row

    def build_transport_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.controls_strip = QHBoxLayout()
        self.controls_strip.setContentsMargins(0, 0, 0, 0)
        row.addLayout(self.controls_strip)
        self.stop_btn = QPushButton(self)
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.setIcon(asset_icon("stop"))
        row.addWidget(self.stop_btn)

        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        row.addSpacing(6)
        row.addWidget(separator)
        row.addSpacing(6)

        self.save_check = QCheckBox("Save", self)
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("Name")
        self.name_edit.setToolTip(
            "What this acquisition is called. Used as the filename when saving, "
            "and as its name in the data panel either way."
        )
        self.dir_btn = QPushButton(self)
        style = self.style()
        if style is not None:
            self.dir_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        row.addWidget(self.save_check)
        row.addWidget(self.name_edit, 1)
        row.addWidget(self.dir_btn)
        return row

    def connect_signals(self) -> None:
        self.program_combo.currentIndexChanged.connect(self.on_program_chosen)
        self.stop_btn.clicked.connect(self.app.runners.stop)
        self.save_check.toggled.connect(lambda checked: self.app.set_save(enabled=checked))
        self.name_edit.textChanged.connect(lambda text: self.app.set_save(name=text))
        self.dir_btn.clicked.connect(self.choose_directory)

        self.app.program_selected.connect(self.on_program_selected)
        self.app.devices_changed.connect(self.refresh_readiness)
        self.app.save_changed.connect(self.on_save_changed)
        self.app.params_written.connect(self.on_params_written)
        runners = self.app.runners
        runners.controls_changed.connect(self.render_controls)
        runners.status.connect(self.show_status)
        runners.picking_changed.connect(self.on_picking_changed)
        runs = self.app.runs
        runs.run_started.connect(lambda _run: self.on_run_started())
        runs.run_status.connect(lambda _run, text: self.show_status(text))
        runs.run_finished.connect(lambda _run: self.on_run_finished())
        runs.run_failed.connect(lambda _run, message: self.on_run_failed(message))
        runs.start_refused.connect(self.on_run_failed)

    def on_program_chosen(self, index: int) -> None:
        key = self.program_combo.itemData(index)
        if key != self.app.selected_program:
            self.app.select_program(key)

    def on_program_selected(self, key: str) -> None:
        index = self.program_combo.findData(key)
        if self.program_combo.currentIndex() != index:
            self.program_combo.blockSignals(True)
            self.program_combo.setCurrentIndex(index)
            self.program_combo.blockSignals(False)
        self.form = ParamForm(
            self.app.params_for(key),
            self,
            cards=True,
            context=FieldContext(library=self.app.library),
        )
        self.form.changed.connect(self.app.state_changed.emit)
        self.form.invalid.connect(self.show_status)
        self.scroll_area.setWidget(self.form)
        self.refresh_readiness()

    def on_params_written(self) -> None:
        """A runner applied a pick to the blocks; show what is there now."""
        if self.form is not None:
            self.form.reload()

    def refresh_readiness(self, *, announce: bool = True) -> None:
        """Enable or disable the controls and say what is missing, if anything.
        ``announce`` is off when a run has just ended, so its outcome is not
        immediately overwritten with "ready"."""
        missing = self.app.blockers()
        self.apply_enabled()
        if missing:
            self.show_status("needs " + ", ".join(missing))
        elif announce and not self.app.runners.running and not self.app.runners.picking:
            self.show_status("ready")

    def show_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")

    def render_controls(self) -> None:
        """Draw the selected program's controls, replacing whatever was there.
        ``clicked`` rather than ``toggled``, so a runner moving its own toggle
        is shown without being mistaken for the user."""
        for control, button, watcher in self.control_buttons:
            control.unwatch(watcher)
            self.controls_strip.removeWidget(button)
            button.deleteLater()
        self.control_buttons = []

        for control in self.app.runners.controls:
            button = QPushButton(self)
            if control.icon is not None:
                button.setIcon(asset_icon(control.icon))
            else:
                button.setText(control.label)
            button.setToolTip(control.tooltip or control.label)
            button.setCheckable(control.checkable)
            button.setChecked(control.checked)
            button.clicked.connect(control.activate)
            watcher = self.make_watcher(control, button)
            control.watch(watcher)
            self.controls_strip.addWidget(button)
            self.control_buttons.append((control, button, watcher))
        self.apply_enabled()

    def make_watcher(self, control: Control, button: QPushButton) -> Callable[[], None]:
        """Keep ``button`` in line with ``control`` when its runner changes it."""

        def watcher() -> None:
            if button.isChecked() != control.checked:
                button.setChecked(control.checked)
            self.apply_enabled()

        return watcher

    def apply_enabled(self) -> None:
        """Off while something is missing or a run is in progress, and off
        whenever the control's runner says so."""
        ready = not self.app.blockers() and not self.app.runners.running
        for control, button, _ in self.control_buttons:
            button.setEnabled(ready and control.enabled)

    def on_picking_changed(self, active: bool) -> None:
        if not active and not self.app.runners.running:
            self.refresh_readiness()

    def choose_directory(self) -> None:
        start = str(self.app.save.folder)
        chosen = QFileDialog.getExistingDirectory(self, "Save acquisitions to", start)
        if chosen:
            self.app.set_save(directory=chosen)

    def on_save_changed(self) -> None:
        """Pull the widgets back in line with the save target, for when a
        restored workspace moved it. Unchanged values are skipped so typing in
        the name field keeps its cursor."""
        save = self.app.save
        if self.name_edit.text() != save.name:
            self.name_edit.setText(save.name)
        if self.save_check.isChecked() != save.enabled:
            self.save_check.setChecked(save.enabled)
        if not save.enabled:
            tooltip = f"Choose where acquisitions are saved. Currently {save.folder}"
        elif not save.filename:
            tooltip = "Name is required when saving is enabled"
        else:
            tooltip = f"Saving to {save.root}_*"
        self.dir_btn.setToolTip(tooltip)
        self.dir_btn.setText(self.folder_name())
        self.refresh_readiness()

    def folder_name(self) -> str:
        """The destination folder's name, readable without hovering: no folder
        chosen means the working directory, which is easy to not expect."""
        folder = self.app.save.folder
        name = folder.name or str(folder)
        return name if len(name) <= 16 else name[:15] + "…"

    def on_run_started(self) -> None:
        self.show_status("acquiring")
        self.set_running_ui(True)

    def on_run_finished(self) -> None:
        self.set_running_ui(self.app.runners.running)
        self.refresh_readiness(announce=False)
        self.show_status("stopped")

    def on_run_failed(self, message: str) -> None:
        self.show_status(f"error - {message}")
        self.set_running_ui(self.app.runners.running)
        QMessageBox.critical(self, "Acquisition Error", message)

    def set_running_ui(self, running: bool) -> None:
        self.apply_enabled()
        self.stop_btn.setEnabled(running)
