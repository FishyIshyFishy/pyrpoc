"""One library entry's metadata, read-only, and its run's note, editable."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.program_output import ProgramOutput

from .grouping import program_display_name


def local_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def add_branch(parent: QTreeWidgetItem, key: str, value: Any) -> None:
    """One key of an encoded dict: nested dicts become branches, the rest leaves."""
    item = QTreeWidgetItem(parent, [key, "" if isinstance(value, dict) else str(value)])
    if isinstance(value, dict):
        for child_key, child in value.items():
            add_branch(item, str(child_key), child)


class DetailsDialog(QDialog):
    """``save_note`` stores the run's note and returns why it could not reach the
    recording's file, or None."""

    def __init__(
        self,
        program_output: ProgramOutput,
        note: str,
        save_note: Callable[[str], str | None],
        parent: QWidget,
    ):
        super().__init__(parent)
        self.save_note = save_note
        self.setWindowTitle(f"{program_output.run_name} · {program_output.name}")
        self.resize(560, 640)

        root = QVBoxLayout(self)
        root.addLayout(self.build_summary(program_output))
        root.addWidget(self.build_tree(program_output), 2)
        root.addWidget(QLabel("Note (one per run, shared by all its outputs)", self))
        self.notes_edit = QPlainTextEdit(note, self)
        self.notes_edit.setPlaceholderText("What happened in this run, and anything to remember.")
        root.addWidget(self.notes_edit, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def build_summary(self, program_output: ProgramOutput) -> QFormLayout:
        rows = [
            ("Name", program_output.run_name),
            ("Output", f"{program_output.name} ({program_output.type_key})"),
            ("Program", program_display_name(program_output.provenance.program_key)),
            ("Origin", program_output.origin.value),
            ("Started", local_time(program_output.provenance.started_at)),
            (
                "File",
                str(program_output.meta_path)
                if program_output.meta_path is not None
                else "Not saved",
            ),
        ]
        form = QFormLayout()
        for label, text in rows:
            value = QLabel(text, self)
            value.setWordWrap(True)
            form.addRow(f"{label}:", value)
        return form

    def build_tree(self, program_output: ProgramOutput) -> QTreeWidget:
        """Parameters and devices, as they were when the recording started."""
        tree = QTreeWidget(self)
        tree.setHeaderLabels(["Setting", "Value"])
        sections = {
            "Parameters": program_output.provenance.parameters,
            "Devices": program_output.provenance.devices,
        }
        for title, values in sections.items():
            section = QTreeWidgetItem(tree, [title, ""])
            for key, value in values.items():
                add_branch(section, str(key), value)
            section.setExpanded(True)
        tree.resizeColumnToContents(0)
        return tree

    def on_save(self) -> None:
        problem = self.save_note(self.notes_edit.toPlainText())
        if problem is not None:
            QMessageBox.warning(
                self,
                "Note Not Written",
                f"The note is kept for this session, but the recording's file "
                f"could not be updated:\n\n{problem}",
            )
            return
        self.accept()
