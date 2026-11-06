"""One recording's metadata, read-only, and its note, editable."""

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

from pyrpoc.core.recording import Recording

from .grouping import program_display_name, recording_title


def local_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def add_branch(parent: QTreeWidgetItem, key: str, value: Any) -> None:
    """One key of an encoded dict: nested dicts become branches, the rest leaves."""
    item = QTreeWidgetItem(parent, [key, "" if isinstance(value, dict) else str(value)])
    if isinstance(value, dict):
        for child_key, child in value.items():
            add_branch(item, str(child_key), child)


class DetailsDialog(QDialog):
    """One recording's metadata and its outputs, read-only, and its note, editable.
    ``save_note`` stores the note and returns why it could not reach the recording's file, or
    None."""

    def __init__(
        self, recording: Recording, save_note: Callable[[str], str | None], parent: QWidget
    ):
        super().__init__(parent)
        self.save_note = save_note
        self.setWindowTitle(recording_title(recording))
        self.resize(560, 640)

        root = QVBoxLayout(self)
        root.addLayout(self.build_summary(recording))
        root.addWidget(self.build_tree(recording), 2)
        root.addWidget(QLabel("Note", self))
        self.notes_edit = QPlainTextEdit(recording.note, self)
        self.notes_edit.setPlaceholderText("What happened in this run, and anything to remember.")
        root.addWidget(self.notes_edit, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def build_summary(self, recording: Recording) -> QFormLayout:
        outputs = ", ".join(
            f"{name} ({output.type_key})" for name, output in recording.outputs.items()
        )
        rows = [
            ("Name", recording_title(recording)),
            ("Program", program_display_name(recording.program_key)),
            ("Started", local_time(recording.started_at)),
            ("Outputs", outputs or "-"),
            ("File", str(recording.location) if recording.location is not None else "Not saved"),
        ]
        form = QFormLayout()
        for label, text in rows:
            value = QLabel(text, self)
            value.setWordWrap(True)
            form.addRow(f"{label}:", value)
        return form

    def build_tree(self, recording: Recording) -> QTreeWidget:
        """Parameters and devices of each part, as they were when the part began."""
        tree = QTreeWidget(self)
        tree.setHeaderLabels(["Setting", "Value"])
        for number, part in enumerate(recording.parts, start=1):
            # One part needs no numbering: most recordings have only one.
            heading = "" if len(recording.parts) == 1 else f"Part {number} "
            sections = {"Parameters": part.parameters, "Devices": part.devices}
            for title, values in sections.items():
                section = QTreeWidgetItem(tree, [f"{heading}{title}", ""])
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
