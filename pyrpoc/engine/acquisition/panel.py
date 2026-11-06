"""The Acquisition panel: choose a program, set its inputs, name the run, and run it."""

from __future__ import annotations

from html import escape
from typing import Any

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.panels import Panel
from pyrpoc.core.program import program_registry
from pyrpoc.core.registry import grouped
from pyrpoc.utils.widgets.control_views import ControlView, build_control_view
from pyrpoc.utils.widgets.grouped_combo import GroupedComboBox
from pyrpoc.utils.widgets.icons import asset_icon
from pyrpoc.utils.widgets.param_form import ParamForm, input_form_section
from pyrpoc.utils.widgets.path_field import PathField

from .earlier_card import EarlierCard
from .host import RunnerHost
from .note import NoteDialog
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
        self.earlier_card: EarlierCard | None = None
        # Each rendered control, its button, and the watcher keeping them in line.
        self.control_views: list[ControlView] = []
        # How the last run ended, so its ending can say so: Stop pressed, or the error it raised.
        self.stop_pressed = False
        self.last_error: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addLayout(self.build_program_row())
        root.addLayout(self.build_transport_row())
        root.addLayout(self.build_save_row())
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
        self.on_note_changed()
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
        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.setToolTip("Stop the running acquisition")
        self.stop_btn.setIcon(asset_icon("stop"))
        row.addWidget(self.stop_btn)
        row.addStretch(1)
        return row

    def build_save_row(self) -> QHBoxLayout:
        """Whether to save, where and under what name, and the next run's note."""
        row = QHBoxLayout()
        self.save_check = QCheckBox("Save", self)
        self.save_check.setToolTip("Save each run into the chosen folder.")
        self.path_field = PathField(self)
        self.path_field.name_edit.setPlaceholderText("name")
        self.path_field.name_edit.setToolTip(
            "What this acquisition is called. Used as the filename when saving, "
            "and as its name in the data panel either way."
        )
        # Small and quiet: a note is written now and then, not every run.
        self.note_btn = QToolButton(self)
        self.note_btn.setAutoRaise(True)
        self.note_btn.setIconSize(QSize(20, 20))
        # Shown pressed in while a note waits for the next run.
        self.note_btn.setCheckable(True)
        row.addWidget(self.save_check)
        row.addWidget(self.path_field, 1)
        row.addWidget(self.note_btn)
        return row

    def connect_signals(self) -> None:
        self.program_combo.currentIndexChanged.connect(self.on_program_chosen)
        self.stop_btn.clicked.connect(self.stop)
        self.save_check.toggled.connect(lambda checked: self.runs.set_save(enabled=checked))
        self.path_field.name_edited.connect(lambda text: self.runs.set_save(name=text))
        self.path_field.folder_clicked.connect(self.choose_directory)
        self.note_btn.clicked.connect(self.edit_note)
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
        self.scroll_area.setWidget(self.build_inputs(key))
        self.refresh_readiness()
        self.state_changed.emit()

    def build_inputs(self, key: str) -> QWidget:
        """The card for what ``key`` reads of earlier recordings, if any, above its program
        inputs."""
        program = program_registry.get(key)
        if self.earlier_card is not None:
            self.earlier_card.detach()
        page = QWidget(self)
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        self.earlier_card = None
        if program.reads:
            # The folder may have changed outside the app since it was last looked at.
            self.runs.refresh_earlier()
            self.earlier_card = EarlierCard(self.runs, program.reads, page)
            column.addWidget(self.earlier_card)
        cards = program.cards or {"Inputs": program.inputs}
        self.form = ParamForm(
            [input_form_section(label, inputs) for label, inputs in cards.items()],
            self.inputs,
            page,
            cards=True,
        )
        self.form.invalid.connect(self.show_status)
        column.addWidget(self.form, 1)
        return page

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
        for view in self.control_views:
            view.close()
            self.controls_strip.removeWidget(view.widget)
            view.widget.deleteLater()
        self.control_views = [build_control_view(control, self) for control in self.host.controls]
        for view in self.control_views:
            self.controls_strip.addWidget(view.widget)
        self.apply_enabled()

    def apply_enabled(self) -> None:
        """Off while something is missing or a run is in progress."""
        missing = self.blockers()
        ready = not missing and not self.host.running
        for view in self.control_views:
            view.widget.setEnabled(ready)
            # A dimmed control says why on hover, so the way forward is one mouse-over away.
            reason = f"\nNeeds {', '.join(missing)}" if missing else ""
            view.widget.setToolTip(view.tooltip + reason)

    def choose_directory(self) -> None:
        folder = self.runs.save.folder
        start = str(folder) if folder is not None else ""
        chosen = QFileDialog.getExistingDirectory(self, "Save acquisitions to", start)
        if chosen:
            self.runs.set_save(directory=chosen)

    def on_save_changed(self) -> None:
        """Bring the save widgets in line with the save target."""
        # For when a restored session moved it.
        save = self.runs.save
        self.path_field.set_name(save.name)
        if self.save_check.isChecked() != save.enabled:
            self.save_check.setChecked(save.enabled)
        folder = save.folder
        if folder is None:
            tooltip = "Choose the folder acquisitions are saved to."
        else:
            tooltip = f"{folder}\nClick to choose another folder."
        name = None if folder is None else short_name(folder.name or str(folder))
        # Dimmed while saving is off: the name then only names the run.
        self.path_field.set_folder(name, tooltip, in_use=save.enabled)
        self.refresh_readiness()

    def edit_note(self) -> None:
        dialog = NoteDialog(self.runs.note, self)
        if dialog.exec():
            self.runs.set_note(dialog.text.strip())
        # The click toggled the button; whether it shows pressed is the note's to say.
        self.on_note_changed()

    def on_note_changed(self) -> None:
        """Show whether a note waits for the next run, and what it says on hover."""
        note = self.runs.note
        self.note_btn.setIcon(asset_icon("note_written" if note else "note"))
        self.note_btn.setChecked(bool(note))
        self.note_btn.setToolTip(
            f"Note for the next run:\n{note}" if note else "Add a note for the next run"
        )

    def stop(self) -> None:
        self.stop_pressed = True
        self.host.stop()

    def on_run_started(self) -> None:
        self.stop_pressed, self.last_error = False, None
        self.show_status("acquiring")
        self.set_running_ui(True)

    def on_run_finished(self) -> None:
        running = self.host.running
        self.set_running_ui(running)
        # A runner may already have started the next run from this same finish.
        if not running:
            self.refresh_readiness(announce=False)
            self.show_ending()

    def show_ending(self) -> None:
        """Say how the run ended. After an error, what the next run needs comes first, since
        the error was already shown in its own box."""
        if self.last_error is None:
            self.show_status("stopped" if self.stop_pressed else "finished")
        elif not self.blockers():
            self.show_status(f"error - {self.last_error}")

    def on_run_failed(self, message: str) -> None:
        self.last_error = message
        self.show_status(f"error - {message}")
        self.set_running_ui(self.host.running)
        QMessageBox.critical(self, "Acquisition Error", message)

    def set_running_ui(self, running: bool) -> None:
        self.apply_enabled()
        self.stop_btn.setEnabled(running)
        # A recording cannot move partway through, so where runs save is fixed while one goes.
        save_fixed = bool(self.runs.live_runs)
        self.save_check.setEnabled(not save_fixed)
        self.path_field.setEnabled(not save_fixed)

    def export_persistence_state(self) -> dict[str, Any]:
        return {"program": self.selected_program}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        """Show the saved program, or the first one if it no longer exists."""
        key = state.get("program")
        self.select_program(key if key in program_registry.entries else program_registry.keys()[0])
