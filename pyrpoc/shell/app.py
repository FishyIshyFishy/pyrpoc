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
from pyrpoc.programs.components import BLOCKS, Point, PointGroup, ScanGroup
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

    pick_armed_changed = pyqtSignal(bool)  # picking turned on or off
    point_acquired = pyqtSignal()          # a pixel became a point in the blocks
    pick_failed = pyqtSignal(str)          # a click could not be placed, and why

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.devices: list[Device] = []
        self.views: list[Any] = []
        self.library = DatasetLibrary()
        self.bridge = RunBridge(self.library, self)

        self.selected_program: str | None = None
        #: Every parameter block that exists, one instance per class. Two
        #: programs declaring ScanGroup are handed the same object, which is
        #: what makes switching modality keep the geometry you set.
        self.blocks = P.BlockStore()
        #: One save target for the session, not one per program: what a run is
        #: called and where it goes has nothing to do with which program runs.
        self.save = SaveTarget()
        #: Whether a click on a display reports a position. Interaction state,
        #: not configuration: it is in ``save``'s category of "a control near
        #: play that is not a parameter of the program", but unlike ``save`` it
        #: has ``library``'s lifetime and is deliberately absent from the
        #: session file. A relaunch that came back armed would be pointing a
        #: hardware trigger at the next stray click.
        self.pick_armed = False

        self.bridge.run_started.connect(lambda: self.state_changed.emit())
        self.bridge.dataset_changed.connect(self.on_dataset_changed)

    # -- programs ----------------------------------------------------------- #

    def select_program(self, key: str) -> None:
        if self.bridge.is_running:
            self.bridge.stop()
        catalog.entry_for(key)  # raises on an unknown key
        self.selected_program = key
        self.params_for(key)    # ensure its blocks exist
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

    # -- picking ------------------------------------------------------------ #

    def set_pick_armed(self, active: bool) -> None:
        """Turn point picking on or off across every view.

        Told to all of them unconditionally rather than to a filtered set:
        ``View.set_picking`` is a no-op by default, so a view with no spatial
        meaning needs no capability flag and this needs no branch. One bad view
        must not leave the rest in a different state from the flag, which is why
        the loop swallows the way ``on_dataset_changed`` does.

        Never emits ``state_changed``. That signal drives the autosave, and this
        is the one piece of state that must not survive a relaunch.
        """
        active = bool(active)
        if active == self.pick_armed:
            return
        self.pick_armed = active
        for view in list(self.views):
            try:
                view.set_picking(active)
            except Exception as exc:  # noqa: BLE001 - one bad view must not stick
                view.last_error = str(exc)
        self.pick_armed_changed.emit(active)

    def on_point_picked(self, dataset_id: str, x: int, y: int) -> None:
        """A display reported a pixel. Turn it into volts and acquire there.

        Disarming happens first, before anything that can fail, so no path out
        of here leaves a live cursor behind.

        The geometry comes from the dataset's provenance rather than the live
        ``ScanGroup``. Blocks are shared and mutable: change the amplitude after
        taking an image and the live block no longer describes the picture being
        clicked, so the volts would point somewhere it never looked.
        """
        if not self.pick_armed:
            return
        self.set_pick_armed(False)

        dataset = self.library.by_id(dataset_id)
        if dataset is None:
            self.pick_failed.emit("that data is no longer open")
            return

        raw = dataset.provenance.parameters.get(P.block_name(ScanGroup))
        if not isinstance(raw, dict):
            self.pick_failed.emit(
                f"{dataset.label} was not acquired with a scan geometry, "
                "so a pixel does not name a position"
            )
            return

        try:
            scan = P.decode_block(ScanGroup, raw)
            fast_v, slow_v = scan.voltage_at(x, y)
        except Exception as exc:  # noqa: BLE001 - reported, not raised: Qt slot
            self.pick_failed.emit(f"could not place that pixel: {exc}")
            return

        self.blocks.get(PointGroup).target = Point(
            fast_v, slow_v, dataset_id, dataset.label, x, y
        )
        self.state_changed.emit()
        self.point_acquired.emit()

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

    def on_dataset_changed(self, dataset) -> None:
        """Refresh the views showing this dataset.

        A hidden view is skipped rather than redrawn into a widget nobody can
        see. It picks the data up on its next refresh -- the next publish it is
        visible for, or the next time its binding changes. Nothing refreshes a
        view on being shown, so one unhidden after a run ended keeps whatever
        it last drew until the next run starts.
        """
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
        #: Connected for every view, including those that never emit. A view
        #: added while picking is on has to arrive already picking, which is why
        #: the state is pushed here rather than only from ``set_pick_armed``.
        view.point_picked.connect(self.on_point_picked)
        view.set_picking(self.pick_armed)
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
        self.blocks.load_dict(raw, BLOCKS)
