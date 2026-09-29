"""The acquisition panel: pick a program, name it, set it up, run it.

One of the three fixed panels -- always present, built once by
app/window.py, not offered under Add. See ``panels/__init__.py`` for how
that differs from the four dataset-rendering panels. Reaches into app/ for
the program catalog, the runner host and the run bridge -- unlike the four
dataset-rendering panels, this one is not held to "must not import app/".

Replaces gui/main_widgets/acquisition_mgr/. The form is generated from the
program's parameter model and writes back into it, so nothing scrapes widgets at
play time -- collect_values is gone.

The transport row starts with the selected program's controls, rendered from
``app.runners``: Start, Continuous and "Acquire at point..." are whatever the
program's runners asked for, and this panel only knows how to draw a
``Button`` and a ``Toggle``. Stop is fixed: every run can be stopped.

The name and the save switch sit in the transport row rather than in the
generated form, because they are not parameters of the program. "Frame" and
"Signal" describe what simulation does; a filename describes what happens to
the result, so it is the same two widgets whichever program is selected and
they belong next to the controls that start the run.
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
    QWidget,
)

from pyrpoc.src.app import catalog
from pyrpoc.src.structs.panel import Panel
from pyrpoc.src.structs.params import FieldContext, ParameterError
from pyrpoc.src.structs.runner import Button, Control, Toggle

from ..components.icons import asset_icon
from ..components.param_form import ParamForm

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.src.app.application import Application


class AcquisitionPanel(Panel):
    display_name = "Acquisition"

    def __init__(self, app: Application, parent: QWidget | None = None):
        super().__init__(parent)
        self.app = app
        self.form: ParamForm | None = None
        # The rendered controls: descriptor, its button, and the watcher
        # that keeps the button in line with it.
        self.control_buttons: list[tuple[Control, QPushButton, Callable[[], None]]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel("Program:", self))
        self.program_combo = QComboBox(self)
        for entry in catalog.CATALOG:
            self.program_combo.addItem(entry.label, entry.key)
        top.addWidget(self.program_combo, 1)
        root.addLayout(top)

        controls = QHBoxLayout()
        style = self.style()
        self.controls_strip = QHBoxLayout()
        self.controls_strip.setContentsMargins(0, 0, 0, 0)
        controls.addLayout(self.controls_strip)
        self.stop_btn = QPushButton(self)
        self.stop_btn.setToolTip("Stop")
        stop_icon = asset_icon("stop")
        if stop_icon is not None:
            self.stop_btn.setIcon(stop_icon)
        elif style is not None:
            self.stop_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        controls.addWidget(self.stop_btn)

        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        controls.addSpacing(6)
        controls.addWidget(separator)
        controls.addSpacing(6)

        self.save_check = QCheckBox("Save", self)
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("Name")
        self.name_edit.setToolTip(
            "What this acquisition is called. Used as the filename when saving, "
            "and as its name in the data panel either way."
        )
        self.dir_btn = QPushButton(self)
        if style is not None:
            self.dir_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        controls.addWidget(self.save_check)
        controls.addWidget(self.name_edit, 1)
        controls.addWidget(self.dir_btn)
        root.addLayout(controls)

        self.status_label = QLabel("Status: idle", self)
        root.addWidget(self.status_label)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        root.addWidget(self.scroll_area, 1)

        self.program_combo.currentIndexChanged.connect(self.on_program_chosen)
        self.stop_btn.clicked.connect(self.app.stop_run)
        self.save_check.toggled.connect(lambda checked: self.app.set_save(enabled=checked))
        self.name_edit.textChanged.connect(lambda text: self.app.set_save(name=text))
        self.dir_btn.clicked.connect(self.choose_directory)

        self.app.program_selected.connect(self.on_program_selected)
        self.app.devices_changed.connect(self.refresh_readiness)
        self.app.save_changed.connect(self.on_save_changed)
        self.app.params_written.connect(self.on_params_written)
        self.app.runners.controls_changed.connect(self.render_controls)
        self.app.runners.status.connect(self.show_status)
        self.app.runners.picking_changed.connect(self.on_picking_changed)
        self.app.bridge.run_started.connect(self.on_run_started)
        self.app.bridge.run_status.connect(self.show_status)
        self.app.bridge.run_finished.connect(self.on_run_finished)
        self.app.bridge.run_failed.connect(self.on_run_failed)

        self.render_controls()
        self.set_running_ui(False)
        self.on_save_changed()
        if self.app.selected_program is None and catalog.CATALOG:
            self.app.select_program(catalog.CATALOG[0].key)
        else:
            self.on_program_selected(self.app.selected_program or "")

    # -- program selection --------------------------------------------------- #

    def on_program_chosen(self, index: int) -> None:
        key = self.program_combo.itemData(index)
        if isinstance(key, str) and key != self.app.selected_program:
            self.app.select_program(key)

    def on_program_selected(self, key: str) -> None:
        if not key:
            return
        index = self.program_combo.findData(key)
        if index >= 0 and self.program_combo.currentIndex() != index:
            self.program_combo.blockSignals(True)
            self.program_combo.setCurrentIndex(index)
            self.program_combo.blockSignals(False)
        self.rebuild_form()
        self.refresh_readiness()

    def rebuild_form(self) -> None:
        params = self.app.current_params()
        if params is None:
            return
        self.form = ParamForm(params, self, context=FieldContext(library=self.app.library))
        self.form.changed.connect(self.app.params_changed.emit)
        self.form.changed.connect(self.app.state_changed.emit)
        self.form.invalid.connect(self.show_status)
        self.scroll_area.setWidget(self.form)

    def on_params_written(self) -> None:
        """Something other than the form wrote the blocks -- a runner applying
        a pick -- so show what is there now."""
        if self.form is not None:
            self.form.reload()

    def refresh_readiness(self, *, announce: bool = True) -> None:
        """Enable or disable the controls, and say what is missing if anything is.

        ``announce`` is off when a run has just ended, so the outcome message is
        not immediately overwritten with "ready".
        """
        if self.app.selected_program is None:
            return
        missing = self.app.blockers()
        running = self.app.bridge.is_running
        self.apply_enabled()
        if missing:
            self.show_status("needs " + ", ".join(missing))
        elif announce and not running and not self.app.runners.picking:
            self.show_status("ready")

    def show_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")

    # -- controls ------------------------------------------------------------- #

    def render_controls(self) -> None:
        """Draw the selected program's controls, replacing whatever was there.

        ``clicked`` rather than ``toggled`` on a toggle, so the runner putting
        its own toggle somewhere is shown without being mistaken for the user.
        """
        for control, button, watcher in self.control_buttons:
            control.unwatch(watcher)
            self.controls_strip.removeWidget(button)
            button.deleteLater()
        self.control_buttons = []

        for control in self.app.runners.controls:
            button = QPushButton(self)
            icon = asset_icon(control.icon) if control.icon else None
            if icon is not None:
                button.setIcon(icon)
            else:
                button.setText(control.label)
            button.setToolTip(control.tooltip or control.label)
            if isinstance(control, Toggle):
                button.setCheckable(True)
                button.setChecked(control.checked)
                button.clicked.connect(lambda checked, c=control: c.toggle(checked))
            elif isinstance(control, Button):
                button.clicked.connect(lambda _checked=False, c=control: c.press())

            watcher = self.make_watcher(control, button)
            control.watch(watcher)
            self.controls_strip.addWidget(button)
            self.control_buttons.append((control, button, watcher))
        self.apply_enabled()

    def make_watcher(self, control: Control, button: QPushButton) -> Callable[[], None]:
        """Keep ``button`` in line with ``control`` when its runner changes it."""

        def watcher() -> None:
            if isinstance(control, Toggle) and button.isChecked() != control.checked:
                button.setChecked(control.checked)
            self.apply_enabled()

        return watcher

    def apply_enabled(self) -> None:
        """One policy for every control: off while something is missing or a
        run is in progress, and off whenever its runner says so."""
        ready = not self.app.blockers() and not self.app.bridge.is_running
        for control, button, _ in self.control_buttons:
            button.setEnabled(ready and control.enabled)

    def on_picking_changed(self, active: bool) -> None:
        if not active and not self.app.bridge.is_running:
            self.refresh_readiness()

    # -- saving --------------------------------------------------------------- #

    def choose_directory(self) -> None:
        start = str(self.app.save.folder)
        chosen = QFileDialog.getExistingDirectory(self, "Save acquisitions to", start)
        if chosen:
            self.app.set_save(directory=chosen)

    def on_save_changed(self) -> None:
        """Pull the widgets back into line with the save target.

        Only needed when something other than these widgets moved it -- a
        restored session. Setting a value it already holds is skipped, so
        typing in the name field does not reset its cursor.
        """
        if self.name_edit.text() != self.app.save.name:
            self.name_edit.setText(self.app.save.name)
        if self.save_check.isChecked() != self.app.save.enabled:
            self.save_check.setChecked(self.app.save.enabled)
        self.dir_btn.setToolTip(self.describe_destination())
        self.dir_btn.setText(self.folder_name())
        self.refresh_readiness()

    def describe_destination(self) -> str:
        if not self.app.save.enabled:
            return f"Choose where acquisitions are saved. Currently {self.app.save.folder}"
        try:
            return f"Saving to {self.app.save.root}_*"
        except ParameterError as exc:
            return str(exc)

    def folder_name(self) -> str:
        """The destination folder's own name, on the button.

        The full path is the tooltip, but which folder has to be readable
        without hovering: no directory chosen means the process's working
        directory, and a run that lands in whatever that happened to be is a
        surprise found later, in the wrong place.
        """
        folder = self.app.save.folder
        name = folder.name or str(folder)
        return name if len(name) <= 16 else name[:15] + "\u2026"

    # -- running -------------------------------------------------------------- #

    def on_run_started(self) -> None:
        self.show_status("acquiring")
        self.set_running_ui(True)

    def on_run_finished(self) -> None:
        self.set_running_ui(False)
        self.refresh_readiness(announce=False)
        self.show_status("stopped")

    def on_run_failed(self, message: str) -> None:
        self.show_status(f"error - {message}")
        self.set_running_ui(False)
        QMessageBox.critical(self, "Acquisition Error", message)

    def set_running_ui(self, running: bool) -> None:
        self.apply_enabled()
        self.stop_btn.setEnabled(running)
