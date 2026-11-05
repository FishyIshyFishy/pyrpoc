"""``SourcePicker``: a dropdown naming one open program output of a type, grouped by the
folder its recording is saved in."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from pyrpoc.core.library import Library
from pyrpoc.core.program_output import ProgramOutput

OutputT = TypeVar("OutputT", bound=ProgramOutput)


def outputs_of_type(library: Library, output_type: type[OutputT]) -> list[OutputT]:
    """Every open output that is an ``output_type``, its daughter types included, newest
    recording first."""
    # isinstance picks one type out of every open output, and narrows them to it.
    return [
        output
        for recording in reversed(library.recordings())
        for output in recording.outputs.values()
        if isinstance(output, output_type)
    ]


def source_text(program_output: ProgramOutput) -> str:
    """One line naming an output: when its recording started, what it was called, and its
    name."""
    recording = program_output.recording
    # No date: a session lasts a day, so a date would repeat on every row.
    started = datetime.fromisoformat(recording.started_at).astimezone().strftime("%H:%M:%S")
    return " · ".join((started, recording.name or recording.program_key, program_output.name))


class SourcePicker(QWidget, Generic[OutputT]):
    """A dropdown naming one open program output of type ``renders``, under the folder its
    recording is saved in. "Latest" follows the newest one."""

    # The chosen program output may have changed; the panel re-reads ``current()``.
    changed = pyqtSignal()

    def __init__(self, renders: type[OutputT], library: Library, parent: QWidget):
        super().__init__(parent)
        self.renders = renders
        self.library = library
        # What each row of the dropdown names: None for "Latest" and for headings.
        self.choices: list[OutputT | None] = []

        # Public so a panel can add its own small controls beside the picker.
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.addWidget(QLabel("Source:", self))
        self.combo = QComboBox(self)
        self.entries = QStandardItemModel(self)
        self.combo.setModel(self.entries)
        self.combo.currentIndexChanged.connect(self.on_chosen)
        self.row.addWidget(self.combo, 1)

        library.subscribe(self.refresh_sources)
        self.refresh_sources()

    def refresh_sources(self) -> None:
        """Rebuild the picker, keeping the current choice while it is open."""
        current = self.chosen()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.choices = []
        self.add_row(QStandardItem("Latest"), None)
        for folder, outputs in self.groups().items():
            self.add_heading(folder)
            for program_output in outputs:
                self.add_row(QStandardItem(source_text(program_output)), program_output)
        if current is not None:
            index = next((i for i, each in enumerate(self.choices) if each is current), 0)
            self.combo.setCurrentIndex(index)
        self.combo.blockSignals(False)
        self.changed.emit()

    def groups(self) -> dict[Path | None, list[OutputT]]:
        """The matching outputs by folder, None for unsaved, the newest first."""
        groups: dict[Path | None, list[OutputT]] = {}
        for program_output in outputs_of_type(self.library, self.renders):
            groups.setdefault(program_output.recording.folder, []).append(program_output)
        return groups

    def add_row(self, item: QStandardItem, choice: OutputT | None) -> None:
        self.entries.appendRow(item)
        self.choices.append(choice)

    def add_heading(self, folder: Path | None) -> None:
        """A bold row naming a folder, which cannot be chosen."""
        item = QStandardItem("Unsaved" if folder is None else folder.name)
        item.setToolTip("Not saved" if folder is None else str(folder))
        # Enabled but not selectable, so the theme does not dim it past reading.
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        font = self.combo.font()
        font.setBold(True)
        item.setFont(font)
        self.add_row(item, None)

    def on_chosen(self, index: int) -> None:
        item = self.entries.item(index)
        if item is not None and not item.isSelectable():
            # Reached by the keyboard or wheel: a heading is always followed by an entry.
            self.combo.setCurrentIndex(index + 1)
            return
        self.changed.emit()

    def chosen(self) -> OutputT | None:
        """The output picked by hand, or None when following the latest."""
        index = self.combo.currentIndex()
        return self.choices[index] if 0 <= index < len(self.choices) else None

    def current(self) -> OutputT | None:
        chosen = self.chosen()
        if chosen is not None:
            return chosen
        candidates = outputs_of_type(self.library, self.renders)
        return candidates[0] if candidates else None
