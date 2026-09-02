"""Application state and the wiring between the ignorant parts.

Ignorant parts still have to be connected, and the only choice is whether the
connecting code lives in one identifiable place or smeared across the parts
meant to stay ignorant. This is that place. When it gets fat, that is a signal
to examine, not something to hide by pushing wiring back into views/ or
programs/.

It replaces AppState plus the five services: instrument -> devices here,
display -> views here, modality -> run/runner, interpreter -> the dataset
notification below, session -> session/.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core import params as P
from pyrpoc.data.io import SaveTarget
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.devices.base import Device
from pyrpoc.devices.registry import device_registry
from pyrpoc.run import claims

from . import catalog
from .run_bridge import RunBridge


class Application(QObject):
    """What exists, how it is configured, and what is running."""

    devices_changed = pyqtSignal()
    views_changed = pyqtSignal()
    program_selected = pyqtSignal(str)
    params_changed = pyqtSignal()
    save_changed = pyqtSignal()
    state_changed = pyqtSignal()          # anything worth autosaving

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.devices: list[Device] = []
        self.views: list[Any] = []
        self.library = DatasetLibrary()
        self.bridge = RunBridge(self.library, self)

        self.selected_program: str | None = None
        self.params_by_program: dict[str, Any] = {}
        #: One save target for the session, not one per program: what a run is
        #: called and where it goes has nothing to do with which program runs.
        self.save = SaveTarget()

        self.bridge.run_started.connect(lambda: self.state_changed.emit())
        self.bridge.dataset_changed.connect(self.on_dataset_changed)

    # -- programs ----------------------------------------------------------- #

    def select_program(self, key: str) -> None:
        if self.bridge.is_running:
            self.bridge.stop()
        catalog.entry_for(key)  # raises on an unknown key
        self.selected_program = key
        self.params_for(key)    # ensure a model exists
        self.program_selected.emit(key)
        self.state_changed.emit()

    def params_for(self, key: str) -> Any:
        """The parameter model for one program, created on first use.

        Kept per program, so switching programs preserves each one's settings.
        """
        if key not in self.params_by_program:
            self.params_by_program[key] = catalog.entry_for(key).program.params()
        return self.params_by_program[key]

    def current_params(self) -> Any | None:
        if self.selected_program is None:
            return None
        return self.params_for(self.selected_program)

    def missing_devices(self, key: str) -> list[str]:
        entry = catalog.entry_for(key)
        return [cls.display_name for cls in claims.missing(list(entry.program.uses), self.devices)]

    # -- saving ------------------------------------------------------------- #

    def set_save(
        self,
        *,
        name: str | None = None,
        directory: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Change part of the save target, leaving the rest alone.

        Only the fields given move, so the launcher's three widgets can each
        write their own without reading the other two back.
        """
        if name is not None:
            self.save.name = str(name)
        if directory is not None:
            self.save.directory = str(directory)
        if enabled is not None:
            self.save.enabled = bool(enabled)
        self.save_changed.emit()
        self.state_changed.emit()

    # -- devices ------------------------------------------------------------ #

    def add_device(self, key: str, **kwargs) -> Device:
        device = device_registry.create(key, **kwargs)
        self.devices.append(device)
        self.devices_changed.emit()
        self.state_changed.emit()
        return device

    def remove_device(self, device: Device) -> None:
        if device not in self.devices:
            return
        self.devices.remove(device)
        self.devices_changed.emit()
        self.state_changed.emit()

    def clear_devices(self) -> None:
        self.devices.clear()
        self.devices_changed.emit()

    # -- views -------------------------------------------------------------- #

    def on_dataset_changed(self, dataset, index: int) -> None:
        """Refresh the views showing this dataset.

        A hidden view is skipped and catches up when it is shown again -- in
        v3.0 it missed those frames permanently, because the interpreter skipped
        anything not currently visible and there was nothing to catch up from.
        """
        del index
        for view in list(self.views):
            if view.dataset() is not dataset:
                continue
            if not getattr(view, "docked_visible", True):
                continue
            try:
                view.refresh()
            except Exception as exc:  # noqa: BLE001 - one bad view must not stop a run
                view.last_error = str(exc)

    def add_view(self, view: Any) -> Any:
        view.attach_library(self.library)
        self.views.append(view)
        self.views_changed.emit()
        self.state_changed.emit()
        return view

    def remove_view(self, view: Any) -> None:
        if view not in self.views:
            return
        self.views.remove(view)
        self.views_changed.emit()
        self.state_changed.emit()

    def clear_views(self) -> None:
        for view in list(self.views):
            self.remove_view(view)

    # -- running ------------------------------------------------------------ #

    def start_run(self, *, continuous: bool = False):
        if self.selected_program is None:
            raise RuntimeError("no program selected")
        entry = catalog.entry_for(self.selected_program)
        params = self.params_for(entry.key)
        return self.bridge.start(
            entry.program(),
            params,
            self.devices,
            continuous=continuous,
            program_key=entry.key,
            save=self.save,
        )

    def stop_run(self) -> None:
        self.bridge.stop()

    # -- persistence -------------------------------------------------------- #

    def params_state(self) -> dict[str, dict]:
        return {key: P.to_dict(model) for key, model in self.params_by_program.items()}

    def load_params_state(self, raw: dict[str, dict]) -> None:
        for key, values in (raw or {}).items():
            try:
                entry = catalog.entry_for(key)
            except KeyError:
                continue
            try:
                self.params_by_program[key] = P.from_dict(entry.program.params, values)
            except Exception:
                self.params_by_program[key] = entry.program.params()
