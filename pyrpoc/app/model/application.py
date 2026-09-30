"""The workbench: what exists, how it is configured, and what is selected.

Panels observe it through the contexts in ``structs/panel.py`` and change it
through its commands; it holds no widgets. Importing the implementation
packages here is what registers every device and program.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.plugins.devices import device_registry
from pyrpoc.plugins.programs import program_registry
from pyrpoc.structs import params as P
from pyrpoc.structs.device import Device, DeviceError
from pyrpoc.structs.params import block_registry
from pyrpoc.structs.saving import SaveTarget

from ..runtime import claims
from ..runtime.library import LIBRARY_LIMIT_BYTES, DataLibrary
from ..runtime.runs import Runs
from .library import LibraryModel
from .runners import Runners

log = logging.getLogger(__name__)


def close_if_open(device: Device) -> None:
    """Close a device's session. A fault while disconnecting is logged, never
    raised: removing a card or quitting must not depend on the hardware."""
    if not device.session_open:
        return
    try:
        device.close_session()
    except DeviceError:
        log.warning("could not cleanly disconnect %s", device.name, exc_info=True)


class Application(QObject):
    """What exists, how it is configured, and what is running."""

    devices_changed = pyqtSignal()
    program_selected = pyqtSignal(str)
    save_changed = pyqtSignal()
    params_written = pyqtSignal()  # blocks changed outside the form
    state_changed = pyqtSignal()  # anything worth autosaving

    def __init__(self) -> None:
        super().__init__()
        self.devices: list[Device] = []
        store = DataLibrary(LIBRARY_LIMIT_BYTES)

        self.selected_program: str | None = None
        # One instance per block class, shared by every program declaring it,
        # so switching modality keeps the settings you made.
        self.blocks = P.BlockStore()
        # One save target for the workspace: where a run goes does not depend
        # on which program runs.
        self.save = SaveTarget()
        self.runs = Runs(store, self)
        self.library = LibraryModel(store, self.runs, self)
        self.runners = Runners(self)

        self.runs.run_started.connect(lambda _run: self.state_changed.emit())
        self.library.auto_purge_changed.connect(lambda _on: self.state_changed.emit())

    def select_program(self, key: str) -> None:
        self.selected_program = key
        self.params_for(key)
        self.runners.attach(key, program_registry.get(key))
        self.program_selected.emit(key)
        self.state_changed.emit()

    def params_for(self, key: str) -> list[P.Group]:
        """The blocks one program declares, in declaration order, created at
        defaults on first request and shared from then on."""
        return [self.blocks.get(cls) for cls in program_registry.get(key).params]

    def blockers(self) -> list[str]:
        """What has to be supplied before the selected program can run. A
        missing name is here too, so the play button says why it is grey
        instead of throwing."""
        missing: list[str] = []
        if self.selected_program is not None:
            uses = list(program_registry.get(self.selected_program).uses)
            missing = [cls.display_name for cls in claims.missing(uses, self.devices)]
        if self.save.enabled and not self.save.filename:
            missing.append("a name to save under")
        return missing

    def set_save(
        self,
        *,
        name: str | None = None,
        directory: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Change only the fields given, so each launcher widget can write its
        own without reading the others back."""
        if name is not None:
            self.save.name = name
        if directory is not None:
            self.save.directory = directory
        if enabled is not None:
            self.save.enabled = enabled
        self.save_changed.emit()
        self.state_changed.emit()

    def add_device(
        self, key: str, instance_id: str | None = None, user_label: str | None = None
    ) -> Device:
        device = device_registry.get(key)(instance_id=instance_id, user_label=user_label)
        self.devices.append(device)
        self.devices_changed.emit()
        self.state_changed.emit()
        return device

    def remove_device(self, device: Device) -> None:
        close_if_open(device)
        self.devices.remove(device)
        self.devices_changed.emit()
        self.state_changed.emit()

    def clear_devices(self) -> None:
        self.close_sessions()
        self.devices.clear()
        self.devices_changed.emit()

    def open_sessions(self) -> list[Device]:
        """Connect every device that keeps a session, returning the ones that
        failed; they stay in the inventory, disconnected, with ``last_error``."""
        failed = [
            device
            for device in self.devices
            if device.holds_session and not device.session_open and not device.try_open_session()
        ]
        self.devices_changed.emit()
        return failed

    def close_sessions(self) -> None:
        for device in self.devices:
            close_if_open(device)

    def params_state(self) -> dict[str, dict]:
        """The block store as a flat state dict, keyed by block class name."""
        return self.blocks.to_dict()

    def load_params_state(self, raw: dict[str, dict]) -> None:
        self.blocks.load_dict(raw, block_registry.entries)
