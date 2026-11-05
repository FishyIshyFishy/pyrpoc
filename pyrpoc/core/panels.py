"""The dock widgets: ``Panel``, ``DataPanel`` with its ``SourcePicker``,
``DevicePanel``, and their registries."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.registry import Registry

from .device import Device, make_instance_id

OutputT = TypeVar("OutputT", bound=ProgramOutput)


class Panel(QWidget):
    """A widget that lives in the dock area."""

    display_name: str = "Panel"
    registry_key: str = "panel"
    # Its own screen state changed, so the session is worth saving.
    state_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.instance_id = make_instance_id(self.registry_key)
        self.user_label: str | None = None

    @property
    def type_key(self) -> str:
        return self.registry_key

    @property
    def title(self) -> str:
        return self.user_label or self.display_name

    def export_persistence_state(self) -> dict[str, Any]:
        return {}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        del state


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


class DataPanel(Panel, Generic[OutputT]):
    """A panel showing one open program output of the type it ``renders``: a source
    picker above a ``body`` the subclass fills through ``draw``."""

    # The one type of program output this panel shows.
    renders: type[OutputT]

    # The heading it is listed under, and its place in the Add menu.
    group: str
    order: int
    # Listed under Add Misc. Panel rather than Add Display: a tool that works
    # on data, not a way to look at it.
    misc: ClassVar[bool] = False

    def __init__(self, library: Library, inputs: ProgramInputs):
        super().__init__()
        self.inputs = inputs
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        self.body = QWidget(self)
        self.source = SourcePicker(self.renders, library, self)
        root.addWidget(self.source)
        root.addWidget(self.body, 1)
        self.shown_output: OutputT | None = None

    def connect_source(self) -> None:
        """Start redrawing on source changes."""
        # Called once the subclass's widgets exist, so a refresh never meets a
        # half-built panel.
        self.source.changed.connect(self.refresh)
        self.refresh()

    def detach(self) -> None:
        """Stop following the library."""
        self.library.unsubscribe(self.source.refresh_sources)

    @property
    def library(self) -> Library:
        return self.source.library

    def program_output(self) -> OutputT | None:
        return self.source.current()

    def refresh(self) -> None:
        program_output = self.program_output()
        self.shown_output = program_output
        if program_output is None:
            self.clear()
            return
        self.draw(program_output)

    def draw(self, program_output: OutputT) -> None:
        """Draw ``program_output``. One that holds nothing yet, as when a run has just started,
        should leave what is on screen, so levels and layout carry over."""
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


data_panel_registry: Registry[DataPanel[Any]] = Registry("DataPanelRegistry", DataPanel)


class DevicePanel(Panel):
    """One device's controls and configuration, docked on their own.
    ``on_change`` reports that its configuration or connection changed."""

    def __init__(self, device: Device, on_change: Callable[[], None]):
        super().__init__()
        # The device's identity, so the dock keeps its place across launches.
        self.instance_id = device.instance_id
        self.device_name = device.name
        self.on_change = on_change

    @property
    def title(self) -> str:
        return self.device_name

    def sync(self) -> None:
        """Re-read the device, which may have changed elsewhere."""

    def set_in_use(self, in_use: bool) -> None:
        """Disable the controls while a run holds the device."""
        self.setEnabled(not in_use)


device_panel_registry: Registry[DevicePanel] = Registry("DevicePanelRegistry", DevicePanel)


def has_panel(device: Device) -> bool:
    return device.registry_key in device_panel_registry.entries
