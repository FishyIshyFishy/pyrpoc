"""Application state, and the wiring between parts that do not know each other.

The connecting code lives here rather than spread across the parts meant to
stay ignorant of each other; when this grows, that is a signal to examine.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.src.panels import DatasetPanel
from pyrpoc.src.structs import params as P
from pyrpoc.src.structs.data import Dataset
from pyrpoc.src.structs.device import Device
from pyrpoc.src.structs.registries import block_registry, device_registry

from . import catalog, claims
from .library import DataLibrary
from .run_bridge import RunBridge
from .runner_host import RunnerHost
from .saving import SaveTarget


class Application(QObject):
    """What exists, how it is configured, and what is running."""

    devices_changed = pyqtSignal()
    panels_changed = pyqtSignal()
    program_selected = pyqtSignal(str)
    save_changed = pyqtSignal()
    params_written = pyqtSignal()  # blocks changed outside the form
    state_changed = pyqtSignal()  # anything worth autosaving

    def __init__(self) -> None:
        super().__init__()
        self.devices: list[Device] = []
        # The added panels. The three fixed panels belong to the window.
        self.panels: list[DatasetPanel] = []
        self.library = DataLibrary()
        self.bridge = RunBridge(self.library, self)

        self.selected_program: str | None = None
        # One instance per block class, shared by every program declaring it,
        # so switching modality keeps the settings you made.
        self.blocks = P.BlockStore()
        # One save target for the session: where a run goes does not depend
        # on which program runs.
        self.save = SaveTarget()
        self.runners = RunnerHost(self)

        self.bridge.run_started.connect(self.state_changed.emit)
        self.bridge.dataset_changed.connect(self.on_dataset_changed)

    def select_program(self, key: str) -> None:
        if self.bridge.is_running:
            self.bridge.stop()
        entry = catalog.entry_for(key)
        self.selected_program = key
        self.params_for(key)
        self.runners.attach(entry.program)
        self.program_selected.emit(key)
        self.state_changed.emit()

    def params_for(self, key: str) -> list[P.Group]:
        """The blocks one program declares, in declaration order, created at
        defaults on first request and shared from then on."""
        return [self.blocks.get(cls) for cls in catalog.entry_for(key).program.params]

    def blockers(self) -> list[str]:
        """What has to be supplied before the selected program can run. A
        missing name is here too, so the play button says why it is grey
        instead of throwing."""
        missing: list[str] = []
        if self.selected_program is not None:
            uses = list(catalog.entry_for(self.selected_program).program.uses)
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
        self.devices.remove(device)
        self.devices_changed.emit()
        self.state_changed.emit()

    def clear_devices(self) -> None:
        self.devices.clear()
        self.devices_changed.emit()

    def on_dataset_changed(self, dataset: Dataset) -> None:
        """Refresh every panel showing this dataset. An added panel exists
        exactly as long as its dock, so there is no unseen panel to skip."""
        for panel in list(self.panels):
            if panel.dataset() is dataset:
                panel.refresh()

    def add_panel(self, panel: DatasetPanel) -> None:
        self.runners.attach_display(panel)
        self.panels.append(panel)
        self.panels_changed.emit()
        self.state_changed.emit()

    def remove_panel(self, panel: DatasetPanel) -> None:
        self.panels.remove(panel)
        panel.detach()
        self.runners.detach_display(panel)
        self.panels_changed.emit()
        self.state_changed.emit()

    def clear_panels(self) -> None:
        for panel in list(self.panels):
            self.remove_panel(panel)

    def start_run(self, *, continuous: bool) -> None:
        if self.selected_program is None:
            raise RuntimeError("no program selected")
        entry = catalog.entry_for(self.selected_program)
        self.params_for(entry.key)  # ensure every declared block exists
        self.bridge.start(
            entry.program(),
            self.blocks,
            self.devices,
            continuous=continuous,
            program_key=entry.key,
            save=self.save,
        )

    def stop_run(self) -> None:
        self.bridge.stop()

    def params_state(self) -> dict[str, dict]:
        """The block store as a flat state dict, keyed by block class name."""
        return self.blocks.to_dict()

    def load_params_state(self, raw: dict[str, dict]) -> None:
        self.blocks.load_dict(raw, block_registry.entries)
