"""The acquisition panel: pick a program, name it, set it up, run it.

Replaces gui/main_widgets/acquisition_mgr/. The form is generated from the
program's parameter model and writes back into it, so nothing scrapes widgets at
play time -- collect_values is gone.

The name and the save switch sit in the transport row rather than in the
generated form, because they are not parameters of the program. "Frame" and
"Signal" describe what simulation does; a filename describes what happens to
the result, so it is the same two widgets whichever program is selected and
they belong next to the button that starts the run.
"""

from __future__ import annotations

from pathlib import Path

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

from pyrpoc.core.errors import ParameterError

from . import catalog
from .app import Application
from .param_form import ParamForm


class LauncherPanel(QWidget):
    def __init__(self, app: Application, parent: QWidget | None = None):
        super().__init__(parent)
        self.app = app
        self.form: ParamForm | None = None

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
        self.start_btn = QPushButton(self)
        self.start_btn.setToolTip("Start")
        self.continuous_btn = QPushButton(self)
        self.continuous_btn.setToolTip("Continuous acquisition")
        self.stop_btn = QPushButton(self)
        self.stop_btn.setToolTip("Stop")
        if style is not None:
            self.start_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.continuous_btn.setIcon(
                style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward)
            )
            self.stop_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        controls.addWidget(self.start_btn)
        controls.addWidget(self.continuous_btn)
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

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        root.addWidget(self.scroll, 1)

        self.program_combo.currentIndexChanged.connect(self.on_program_chosen)
        self.start_btn.clicked.connect(lambda: self.start(continuous=False))
        self.continuous_btn.clicked.connect(lambda: self.start(continuous=True))
        self.stop_btn.clicked.connect(self.app.stop_run)
        self.save_check.toggled.connect(lambda checked: self.app.set_save(enabled=checked))
        self.name_edit.textChanged.connect(lambda text: self.app.set_save(name=text))
        self.dir_btn.clicked.connect(self.choose_directory)

        self.app.program_selected.connect(self.on_program_selected)
        self.app.devices_changed.connect(self.refresh_readiness)
        self.app.save_changed.connect(self.on_save_changed)
        self.app.bridge.run_started.connect(self.on_run_started)
        self.app.bridge.run_status.connect(lambda text: self.status_label.setText(f"Status: {text}"))
        self.app.bridge.run_finished.connect(self.on_run_finished)
        self.app.bridge.run_failed.connect(self.on_run_failed)

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
        self.form = ParamForm(params, self)
        self.form.changed.connect(self.app.params_changed.emit)
        self.form.changed.connect(self.app.state_changed.emit)
        self.form.invalid.connect(lambda text: self.status_label.setText(f"Status: {text}"))
        self.scroll.setWidget(self.form)

    def refresh_readiness(self, *, announce: bool = True) -> None:
        """Enable or disable play, and say what is missing if anything is.

        ``announce`` is off when a run has just ended, so the outcome message is
        not immediately overwritten with "ready".
        """
        if self.app.selected_program is None:
            return
        missing = self.blockers()
        running = self.app.bridge.is_running
        self.start_btn.setEnabled(not missing and not running)
        self.continuous_btn.setEnabled(not missing and not running)
        if missing:
            self.status_label.setText("Status: needs " + ", ".join(missing))
        elif announce and not running:
            self.status_label.setText("Status: ready")

    def blockers(self) -> list[str]:
        """What has to be supplied before a run can start.

        The empty name is in here rather than left to fail at play time: the
        runner raises on it, and a play button that throws is worse than one
        that says why it is grey.
        """
        key = self.app.selected_program
        missing = self.app.missing_devices(key) if key else []
        if self.app.save.enabled and not self.app.save.filename:
            missing = missing + ["a name to save under"]
        return missing

    # -- saving --------------------------------------------------------------- #

    def choose_directory(self) -> None:
        start = self.app.save.directory.strip() or str(Path.cwd())
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
        self.refresh_readiness()

    def describe_destination(self) -> str:
        if not self.app.save.enabled:
            return "Choose where acquisitions are saved"
        try:
            return f"Saving to {self.app.save.root}_*"
        except ParameterError as exc:
            return str(exc)

    # -- running -------------------------------------------------------------- #

    def start(self, *, continuous: bool) -> None:
        try:
            self.app.start_run(continuous=continuous)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.status_label.setText(f"Status: error - {exc}")
            QMessageBox.critical(self, "Acquisition Error", str(exc))

    def on_run_started(self) -> None:
        self.status_label.setText("Status: acquiring")
        self.set_running_ui(True)

    def on_run_finished(self, frame_count: int) -> None:
        self.set_running_ui(False)
        self.refresh_readiness(announce=False)
        self.status_label.setText(f"Status: stopped ({frame_count} frames)")

    def on_run_failed(self, message: str) -> None:
        self.status_label.setText(f"Status: error - {message}")
        self.set_running_ui(False)
        QMessageBox.critical(self, "Acquisition Error", message)

    def set_running_ui(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.continuous_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
