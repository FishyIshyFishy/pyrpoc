"""Application state and the wiring between the ignorant parts.

Ignorant parts still have to be connected, and the only choice is whether the
connecting code lives in one identifiable place or smeared across the parts
meant to stay ignorant. This is that place. When it gets fat, that is a signal
to examine, not something to hide by pushing wiring back into panels/ or
programs/.

It replaces AppState plus the five services: instrument -> devices here,
display -> panels here, modality -> executor.py, interpreter -> the dataset
notification below, session -> session.py.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.src.structs import params as P
from pyrpoc.src.structs.device import Device
from pyrpoc.src.structs.registries import block_registry, device_registry

from . import catalog
from . import claims
from .library import DataLibrary
from .run_bridge import RunBridge
from .runner_host import RunnerHost
from .saving import SaveTarget


class Application(QObject):
    """What exists, how it is configured, and what is running."""

    devices_changed = pyqtSignal()
    panels_changed = pyqtSignal()
    program_selected = pyqtSignal(str)
    params_changed = pyqtSignal()
    save_changed = pyqtSignal()
    params_written = pyqtSignal()         # blocks changed outside the form
    state_changed = pyqtSignal()          # anything worth autosaving

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.devices: list[Device] = []
        #: The added panels -- image_2d, overlay, mask_editor, spectrum
        #: instances the Add menu has created. The three fixed panels
        #: (acquisition, devices, data library) are not in here: they are
        #: built once by app/window.py and never removed.
        self.panels: list[Any] = []
        self.library = DataLibrary()
        self.bridge = RunBridge(self.library, self)

        self.selected_program: str | None = None
        #: Every parameter block that exists, one instance per class. Two
        #: programs declaring the same block are handed the same object, which is
        #: what makes switching modality keep the settings you made.
        self.blocks = P.BlockStore()
        #: One save target for the session, not one per program: what a run is
        #: called and where it goes has nothing to do with which program runs.
        self.save = SaveTarget()
        #: The selected program's entry points. The app hosts them and knows
        #: nothing else about them.
        self.runners = RunnerHost(self)

        self.bridge.run_started.connect(lambda: self.state_changed.emit())
        self.bridge.dataset_changed.connect(self.on_dataset_changed)

    # -- programs ----------------------------------------------------------- #

    def select_program(self, key: str) -> None:
        if self.bridge.is_running:
            self.bridge.stop()
        catalog.entry_for(key)  # raises on an unknown key
        self.selected_program = key
        self.params_for(key)    # ensure its blocks exist
        self.runners.attach(catalog.entry_for(key).program)
        self.program_selected.emit(key)
        self.state_changed.emit()

    def params_for(self, key: str) -> list:
        """The blocks one program declares, in the order it declared them.

        Created at defaults on first request and shared from then on: a block
        two programs both declare is one object, so a value set under one
        modality is already set under the other.
        """
        declared = catalog.entry_for(key).program.params
        return [self.blocks.get(cls) for cls in declared]

    def current_params(self) -> list | None:
        if self.selected_program is None:
            return None
        return self.params_for(self.selected_program)

    def missing_devices(self, key: str) -> list[str]:
        entry = catalog.entry_for(key)
        return [cls.display_name for cls in claims.missing(list(entry.program.uses), self.devices)]

    def blockers(self) -> list[str]:
        """What has to be supplied before the selected program can run.

        The empty name is in here rather than left to fail at play time: the
        executor raises on it, and a play button that throws is worse than one
        that says why it is grey.
        """
        key = self.selected_program
        missing = self.missing_devices(key) if key else []
        if self.save.enabled and not self.save.filename:
            missing = missing + ["a name to save under"]
        return missing

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

    # -- panels --------------------------------------------------------------- #

    def on_dataset_changed(self, dataset) -> None:
        """Refresh every panel showing this dataset.

        Unconditionally, because an added panel exists exactly as long as its
        dock does: closing one deletes it, so there is no open-but-unseen
        panel to skip redrawing.
        """
        for panel in list(self.panels):
            if panel.dataset() is not dataset:
                continue
            try:
                panel.refresh()
            except Exception as exc:  # noqa: BLE001 - one bad panel must not stop a run
                panel.last_error = str(exc)

    def add_panel(self, panel: Any) -> Any:
        panel.attach_library(self.library)
        self.runners.attach_display(panel)
        self.panels.append(panel)
        self.panels_changed.emit()
        self.state_changed.emit()
        return panel

    def remove_panel(self, panel: Any) -> None:
        if panel not in self.panels:
            return
        self.panels.remove(panel)
        self.runners.detach_display(panel)
        self.panels_changed.emit()
        self.state_changed.emit()

    def clear_panels(self) -> None:
        for panel in list(self.panels):
            self.remove_panel(panel)

    # -- running ------------------------------------------------------------ #

    def start_run(self, *, continuous: bool = False):
        if self.selected_program is None:
            raise RuntimeError("no program selected")
        entry = catalog.entry_for(self.selected_program)
        self.params_for(entry.key)  # ensure every declared block exists
        return self.bridge.start(
            entry.program(),
            self.blocks,
            self.devices,
            continuous=continuous,
            program_key=entry.key,
            save=self.save,
        )

    def stop_run(self) -> None:
        self.bridge.stop()

    # -- persistence -------------------------------------------------------- #

    def params_state(self) -> dict[str, dict]:
        """The block store as a flat state dict, keyed by block class name."""
        return self.blocks.to_dict()

    def load_params_state(self, raw: dict[str, dict]) -> None:
        self.blocks.load_dict(raw, block_registry.entries)
