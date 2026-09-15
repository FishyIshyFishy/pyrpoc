"""Arm a crosshair, click a display, park the galvos there and acquire.

Split out of ``Application`` because it is a complete feature on its own --
arm, click, convert, write, signal -- that only needs a dataset library and
the shared block store, not the rest of what an ``Application`` is (devices,
runs, save target). ``Application`` still owns the one thing this cannot: the
list of open views, which is who gets told when arming changes.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from pyrpoc.core import params as P
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.programs.components import Point, PointGroup, ScanGroup


class PickingController(QObject):
    """Whether picking is armed, and what an incoming pick does."""

    armed_changed = pyqtSignal(bool)       # picking turned on or off
    point_acquired = pyqtSignal()          # a pixel became a point in the blocks
    pick_failed = pyqtSignal(str)          # a click could not be placed, and why

    def __init__(
        self,
        library: DatasetLibrary,
        blocks: P.BlockStore,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.library = library
        self.blocks = blocks
        self.armed = False

    def set_armed(self, active: bool) -> None:
        active = bool(active)
        if active == self.armed:
            return
        self.armed = active
        self.armed_changed.emit(active)

    def on_point_picked(self, dataset_id: str, x: int, y: int) -> None:
        """A display reported a pixel. Turn it into volts and acquire there.

        Disarming happens first, before anything that can fail, so no path out
        of here leaves a live cursor behind.

        The geometry comes from the dataset's provenance rather than the live
        ``ScanGroup``. Blocks are shared and mutable: change the amplitude after
        taking an image and the live block no longer describes the picture being
        clicked, so the volts would point somewhere it never looked.
        """
        if not self.armed:
            return
        self.set_armed(False)

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
        self.point_acquired.emit()
