"""The dock widgets: ``Panel``, ``DataPanel`` with its ``SourcePicker``,
``DevicePanel``, and their registries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.core.program_output import Data, ProgramOutput
from pyrpoc.core.registry import Registry

from .device import Device, make_instance_id

D = TypeVar("D", bound=Data)


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


class SourcePicker(QWidget, Generic[D]):
    """A dropdown naming one open program output of kind ``renders``. "Latest" follows
    the newest one."""

    # The chosen program output may have changed; the panel re-reads ``current()``.
    changed = pyqtSignal()

    def __init__(self, renders: type[D], library: Library, parent: QWidget):
        super().__init__(parent)
        self.renders = renders
        self.library = library
        self._follow_latest = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Source:", self))
        self.combo = QComboBox(self)
        self.combo.currentIndexChanged.connect(self.on_chosen)
        layout.addWidget(self.combo, 1)

        library.subscribe(self.refresh_sources)
        self.refresh_sources()

    def refresh_sources(self) -> None:
        """Rebuild the picker, keeping the current choice where possible."""
        current = self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("Latest", None)
        for program_output in self.library.matching(self.renders):
            self.combo.addItem(program_output.label, program_output.id)
        if not self._follow_latest:
            self.combo.setCurrentIndex(max(self.combo.findData(current), 0))
        self.combo.blockSignals(False)
        self.changed.emit()

    def on_chosen(self, _index: int) -> None:
        self._follow_latest = self.combo.currentData() is None
        self.changed.emit()

    def current(self) -> ProgramOutput[D] | None:
        chosen = self.combo.currentData()
        if chosen is None:
            candidates = self.library.matching(self.renders)
            return candidates[0] if candidates else None
        return self.library.by_id(chosen, self.renders)


class DataPanel(Panel, Generic[D]):
    """A panel showing one open program output of the kind it ``renders``: a source
    picker above a ``body`` the subclass fills through ``show_value``."""

    # The one kind of data this panel shows.
    renders: type[D]

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
        self.shown_output: ProgramOutput[D] | None = None
        self.shown_value: D | None = None

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

    def program_output(self) -> ProgramOutput[D] | None:
        return self.source.current()

    def refresh(self) -> None:
        program_output = self.program_output()
        if program_output is None:
            self.shown_output = self.shown_value = None
            self.clear()
            return
        value = program_output.value
        # A program output with no value yet, as when a run has just started, leaves the
        # last value on screen so levels and layout carry over.
        # Library changes re-announce the source even when nothing new arrived.
        if value is None or (program_output is self.shown_output and value is self.shown_value):
            return
        self.shown_output, self.shown_value = program_output, value
        self.show_value(program_output, value)

    def show_value(self, program_output: ProgramOutput[D], value: D) -> None:
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
