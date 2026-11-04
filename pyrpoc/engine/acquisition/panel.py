"""The Acquisition panel: choose a program, set its inputs, name the run, and run it."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
)

from pyrpoc.core.panels import Panel
from pyrpoc.core.program import program_registry
from pyrpoc.core.registry import grouped
from pyrpoc.core.runner import Control
from pyrpoc.utils.widgets.grouped_combo import GroupedComboBox
from pyrpoc.utils.widgets.icons import asset_icon
from pyrpoc.utils.widgets.param_form import ParamForm, input_form_section

from .host import RunnerHost
from .program_inputs import ProgramInputsStore
from .runs import Runs

# Status dot colours, by what the status asks of the user: wait, act, or look.
_DOT_READY = "#3ecf8e"
_DOT_NEEDS = "#f2c744"
_DOT_BUSY = "#4f8df5"
_DOT_ERROR = "#e5484d"


def status_dot(text: str) -> str:
    """The colour of the dot that leads a status line."""
    if text.startswith("needs"):
        return _DOT_NEEDS
    if text.startswith("error"):
        return _DOT_ERROR
    if text.startswith("acquiring"):
        return _DOT_BUSY
    return _DOT_READY


def short_name(name: str) -> str:
    """``name`` cut to fit a narrow panel."""
    return name if len(name) <= 16 else name[:15] + "…"


class AcquisitionPanel(Panel):
    """Shows one program at a time: its runners' controls, its inputs, and where
    its runs save. Which program it shows is its own screen state."""

    display_name = "Acquisition"

    def __init__(self, runs: Runs, inputs: ProgramInputsStore):
        super().__init__()
        self.runs = runs
        self.inputs = inputs
        self.host = RunnerHost(runs, inputs, self)
        self.selected_program: str | None = None
        self.form: ParamForm | None = None
        # Each rendered control, its button, and the watcher keeping them in line.
        self.control_buttons: list[tuple[Control, QPushButton, Callable[[], None]]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addLayout(self.build_program_row())
        root.addLayout(self.build_transport_row())
        root.addLayout(self.build_save_row())
        root.addWidget(self.build_note_box())
        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)
        # Wrapped text would otherwise hold the panel wider than its dock.
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        root.addWidget(self.status_label)
        self.show_status("idle")
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(self.scroll_area, 1)

        self.connect_signals()
        self.render_controls()
        self.set_running_ui(False)
        self.on_save_changed()
        self.select_program(program_registry.keys()[0])

    def build_program_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Program:", self))
        self.program_combo = GroupedComboBox(self)
        for heading, keys in grouped(program_registry).items():
            self.program_combo.add_group(
                heading, [(program_registry.get(key).display_name, key) for key in keys]
            )
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
        row.addStretch(1)
        return row

    def build_save_row(self) -> QHBoxLayout:
        """Whether and where to save, on its own row so a narrow panel does not squeeze the
        name field."""
        row = QHBoxLayout()
        self.save_check = QCheckBox("Save", self)
        self.save_check.setToolTip("Save each run into the chosen folder.")
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
        self.name_edit.setMinimumWidth(80)
        row.addWidget(self.name_edit, 1)
        row.addWidget(self.dir_btn)
        return row

    def build_note_box(self) -> QPlainTextEdit:
        """What the user writes before pressing run, saved with that run and then cleared."""
        self.note_edit = QPlainTextEdit(self)
        self.note_edit.setPlaceholderText(
            "Note for the next run: what you're about to do, and why. "
            "Saved with that run, then cleared."
        )
        self.note_edit.setToolTip(
            "Saved with the run's files, and gathered with the folder's other notes in "
            "experiment.txt. "
            "Edit it afterwards from the run's details in the Data Library."
        )
        self.note_edit.setTabChangesFocus(True)
        lines = self.note_edit.fontMetrics().lineSpacing()
        self.note_edit.setFixedHeight(3 * lines + 14)
        return self.note_edit

    def connect_signals(self) -> None:
        self.program_combo.currentIndexChanged.connect(self.on_program_chosen)
        self.stop_btn.clicked.connect(self.host.stop)
        self.save_check.toggled.connect(lambda checked: self.runs.set_save(enabled=checked))
        self.name_edit.textChanged.connect(lambda text: self.runs.set_save(name=text))
        self.dir_btn.clicked.connect(self.choose_directory)
        self.note_edit.textChanged.connect(lambda: self.runs.set_note(self.note_edit.toPlainText()))
        self.runs.note_changed.connect(self.on_note_changed)

        self.runs.inventory.changed.connect(self.refresh_readiness)
        self.runs.save_changed.connect(self.on_save_changed)
        self.inputs.subscribe(self.on_input_changed)
        self.host.controls_changed.connect(self.render_controls)
        self.host.status.connect(self.show_status)
        runs = self.runs.events
        runs.run_started.connect(lambda _run: self.on_run_started())
        runs.run_status.connect(lambda _run, text: self.show_status(text))
        runs.run_finished.connect(lambda _run: self.on_run_finished())
        runs.run_failed.connect(lambda _run, message: self.on_run_failed(message))
        runs.start_refused.connect(self.on_run_failed)

    def on_program_chosen(self, index: int) -> None:
        key = self.program_combo.itemData(index)
        if key != self.selected_program:
            self.select_program(key)

    def select_program(self, key: str) -> None:
        """Show ``key``: its runners' controls and its inputs."""
        self.selected_program = key
        self.host.attach(key, program_registry.get(key))
        index = self.program_combo.findData(key)
        if self.program_combo.currentIndex() != index:
            self.program_combo.blockSignals(True)
            self.program_combo.setCurrentIndex(index)
            self.program_combo.blockSignals(False)
        self.form = ParamForm(
            [input_form_section(each) for each in program_registry.get(key).sections],
            self.inputs,
            self,
            cards=True,
        )
        self.form.invalid.connect(self.show_status)
        self.scroll_area.setWidget(self.form)
        self.refresh_readiness()
        self.state_changed.emit()

    def on_input_changed(self, key: str) -> None:
        """Show a value set elsewhere, such as a mask drawn in a Mask Editor."""
        del key
        # Every field, not just ``key``'s: an editor may show other inputs beside its own, as
        # Modulation's lines show the drawn masks.
        if self.form is not None:
            self.form.reload()

    def blockers(self) -> list[str]:
        """What the shown program still needs before it can run."""
        if self.selected_program is None:
            return []
        return self.runs.blockers(self.selected_program)

    def refresh_readiness(self, *, announce: bool = True) -> None:
        """Enable or disable the controls and say what is missing, if anything."""
        # ``announce`` is off when a run has just ended, so its outcome is not immediately
        # overwritten with "ready".
        missing = self.blockers()
        self.apply_enabled()
        if missing:
            self.show_status("needs " + ", ".join(missing))
        elif announce and not self.host.running:
            self.show_status("ready")

    def show_status(self, text: str) -> None:
        self.status_label.setText(
            f'<span style="color:{status_dot(text)}">●</span>&nbsp;{escape(text)}'
        )

    def render_controls(self) -> None:
        """Draw the shown program's controls, replacing whatever was there."""
        # ``clicked`` rather than ``toggled``, so a runner moving its own toggle is shown without
        # being mistaken for the user.
        for control, button, watcher in self.control_buttons:
            control.unwatch(watcher)
            self.controls_strip.removeWidget(button)
            button.deleteLater()
        self.control_buttons = []

        for control in self.host.controls:
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
        missing = self.blockers()
        ready = not missing and not self.host.running
        for control, button, _ in self.control_buttons:
            button.setEnabled(ready and control.enabled)
            # A dimmed button says why on hover, so the way forward is one mouse-over away.
            reason = f"\nNeeds {', '.join(missing)}" if missing else ""
            button.setToolTip((control.tooltip or control.label) + reason)

    def choose_directory(self) -> None:
        folder = self.runs.save.folder
        start = str(folder) if folder is not None else ""
        chosen = QFileDialog.getExistingDirectory(self, "Save acquisitions to", start)
        if chosen:
            self.runs.set_save(directory=chosen)

    def on_save_changed(self) -> None:
        """Bring the save widgets in line with the save target."""
        # For when a restored session moved it. Unchanged values are skipped so typing in the name
        # field keeps its cursor.
        save = self.runs.save
        if self.name_edit.text() != save.name:
            self.name_edit.setText(save.name)
        if self.save_check.isChecked() != save.enabled:
            self.save_check.setChecked(save.enabled)
        folder = save.folder
        if folder is None:
            self.dir_btn.setText("Folder…")
            self.dir_btn.setToolTip("Choose the folder acquisitions are saved to.")
        else:
            self.dir_btn.setText(short_name(folder.name or str(folder)))
            self.dir_btn.setToolTip(f"Saving to {folder}\nClick to choose another folder.")
        self.refresh_readiness()

    def on_note_changed(self) -> None:
        """Show the note a run took being cleared."""
        # Unchanged text is skipped so typing keeps its cursor.
        if self.note_edit.toPlainText() != self.runs.note:
            self.note_edit.setPlainText(self.runs.note)

    def on_run_started(self) -> None:
        self.show_status("acquiring")
        self.set_running_ui(True)

    def on_run_finished(self) -> None:
        running = self.host.running
        self.set_running_ui(running)
        # A runner may already have started the next run from this same finish.
        if not running:
            self.refresh_readiness(announce=False)
            self.show_status("stopped")

    def on_run_failed(self, message: str) -> None:
        self.show_status(f"error - {message}")
        self.set_running_ui(self.host.running)
        QMessageBox.critical(self, "Acquisition Error", message)

    def set_running_ui(self, running: bool) -> None:
        self.apply_enabled()
        self.stop_btn.setEnabled(running)

    def export_persistence_state(self) -> dict[str, Any]:
        return {"program": self.selected_program}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        """Show the saved program, or the first one if it no longer exists."""
        key = state.get("program")
        self.select_program(key if key in program_registry.entries else program_registry.keys()[0])
