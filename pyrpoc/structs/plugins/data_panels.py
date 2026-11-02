"""Widgets that live in the dock area: the data-panel and device-panel contracts.

A ``Panel`` has identity and a title. A ``DataPanel`` shows one open dataset
of the one kind it renders, chosen with a ``SourcePicker``; data-panel plugins
subclass it and register in ``data_panel_registry``. A ``DevicePanel`` holds
one device's controls and registers in ``device_panel_registry`` under its
device's key. The only Qt file in structs, since a panel is a widget.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from pyrpoc.structs.data_library.data import Data
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.library import Library
from pyrpoc.structs.registry import Registry

from .devices import Device, make_instance_id

D = TypeVar("D", bound=Data)


class Panel(QWidget):
    """A widget that lives in the dock area."""

    display_name: str = "Panel"
    registry_key: str = "panel"

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
    """Names one open dataset of kind ``renders``. "Latest" follows the newest
    one, so a new run shows up without a click."""

    # The chosen dataset may have changed; the panel re-reads ``current()``.
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
        for dataset in self.library.matching(self.renders):
            self.combo.addItem(dataset.label, dataset.id)
        if not self._follow_latest:
            self.combo.setCurrentIndex(max(self.combo.findData(current), 0))
        self.combo.blockSignals(False)
        self.changed.emit()

    def on_chosen(self, _index: int) -> None:
        self._follow_latest = self.combo.currentData() is None
        self.changed.emit()

    def current(self) -> Dataset[D] | None:
        chosen = self.combo.currentData()
        if chosen is None:
            candidates = self.library.matching(self.renders)
            return candidates[0] if candidates else None
        return self.library.by_id(chosen, self.renders)


class DataPanel(Panel, Generic[D]):
    """A source picker above a ``body`` the subclass fills. ``refresh`` runs
    whenever the chosen dataset or its value changes, and hands the value to
    ``show_value``, which updates the widgets it already has.

    A chosen dataset with no value yet, as when a run has just started,
    leaves the last value on screen: the next run replaces the picture rather
    than blanking it, and levels and layout carry over. Only having no
    dataset at all calls ``clear``.

    Every one can be told a pixel is wanted and can emit one; panels that
    cannot provide pixels keep the no-op, so the host routes to all of them.
    """

    # The one kind of data this panel shows.
    renders: type[D]

    # The heading it is listed under, and its place in the Add menu.
    group: str
    order: int
    # Listed under Add Misc. Panel rather than Add Display: a tool that works
    # on data, not a way to look at it.
    misc: ClassVar[bool] = False

    # A ``Dataset[Pixel]`` the user clicked while picking.
    picked = pyqtSignal(object)

    def __init__(self, library: Library):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        self.body = QWidget(self)
        self.source = SourcePicker(self.renders, library, self)
        root.addWidget(self.source)
        root.addWidget(self.body, 1)
        self.shown_dataset: Dataset[D] | None = None
        self.shown_value: D | None = None

    def connect_source(self) -> None:
        """Start redrawing on source changes. Called by subclasses once their
        widgets exist, so a refresh never meets a half-built panel."""
        self.source.changed.connect(self.refresh)
        self.refresh()

    def detach(self) -> None:
        """Stop following the library; called when the panel is removed."""
        self.library.unsubscribe(self.source.refresh_sources)

    @property
    def library(self) -> Library:
        return self.source.library

    def dataset(self) -> Dataset[D] | None:
        return self.source.current()

    def refresh(self) -> None:
        dataset = self.dataset()
        if dataset is None:
            self.shown_dataset = self.shown_value = None
            self.clear()
            return
        value = dataset.value
        # Library changes re-announce the source even when nothing new arrived.
        if value is None or (dataset is self.shown_dataset and value is self.shown_value):
            return
        self.shown_dataset, self.shown_value = dataset, value
        self.show_value(dataset, value)

    def show_value(self, dataset: Dataset[D], value: D) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def set_picking(self, picking: bool) -> None:
        """Offer the user a pixel to click, or stop offering."""
        del picking


data_panel_registry: Registry[DataPanel[Any]] = Registry("DataPanelRegistry", DataPanel)


class DevicePanel(Panel):
    """One device's controls and configuration, docked on their own. A device
    has at most one open; closing it leaves the device in the inventory.

    ``on_change`` reports that the device's configuration or connection
    changed, so the inventory saves it and every view re-reads it.
    """

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
        """Re-read the device, which may have been connected or disconnected
        from elsewhere, such as the inventory."""

    def set_in_use(self, in_use: bool) -> None:
        """A run holds the device: controls must not touch it meanwhile."""
        self.setEnabled(not in_use)


device_panel_registry: Registry[DevicePanel] = Registry("DevicePanelRegistry", DevicePanel)


def has_panel(device: Device) -> bool:
    return device.registry_key in device_panel_registry.entries
