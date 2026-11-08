"""``FakeLoop``: a loop over a ``FakeDish`` on the wall clock, logging the events a real loop logs
and taking changes only between steps, as a real loop does. It runs in the order the panel
shows, by the same rule."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy as np

from pyrpoc.plugins.program_io.outputs import (
    FocusChecked,
    FocusingAtUnit,
    FrameTag,
    LoopEvent,
    LoopFinished,
    LoopStarted,
    LoopView,
    Measured,
    MovingToUnit,
    Placement,
    RoutineEnded,
    RunNowAsked,
    ScanningUnit,
    StepDue,
    Unit,
    UnitJoined,
    UnitLeft,
    UnitPaused,
    UnitRefused,
    UnitResumed,
    UnitScanned,
)
from pyrpoc.plugins.program_io.outputs.loop_view import LoopFold

from ..feed import Backdrop, UnitFrame
from ..state.outline import fits, middle
from ..state.phase import Phase
from ..state.routines import RoutineKey, RoutineStatus, routine_statuses
from ..state.run_order import next_to_run
from ..state.snapshot import Snapshot, snapshot_of
from .fake_dish import CHANNELS, FIELD_PX, FIELD_UM, OVERVIEW_UM_PER_PX, FakeDish

FOCUS_S, SCAN_S = 1.4, 2.2
# A drawn unit follows the cell nearest its middle when one is this close, in µm.
FOLLOW_UM = 30.0
# The share of scans that do not find the unit where it was.
LOST_SHARE = 0.06


@dataclass(frozen=True)
class FakeRoutine:
    """Scan every ``every_s`` seconds, ``scans`` times, checking focus every ``focus_every``
    scans."""

    name: str
    every_s: float
    scans: int
    focus_every: int


ROUTINES = (FakeRoutine("monitor", 20.0, 40, 4), FakeRoutine("treat", 45.0, 8, 2))


@dataclass
class Busy:
    """The step running now: whose, which phase, when the phase ends, when it was due, and where
    the stage went."""

    key: RoutineKey
    phase: Phase
    ends_s: float
    due_s: float
    x_um: float
    y_um: float


class FakeLoop:
    def __init__(self, dish: FakeDish, units: int):
        self.dish = dish
        self.started = time.monotonic()
        self.fold = LoopFold()
        self.told: tuple[LoopView, tuple[RoutineStatus, ...]] | None = None
        self.order: tuple[RoutineKey, ...] = ()
        # Changes asked for, taken between steps.
        self.inbox: list[Callable[[float], None]] = []
        self.busy: Busy | None = None
        self.cells: dict[int, int | None] = {}
        self.scans: dict[RoutineKey, int] = {}
        self.asked: dict[int, int] = {}
        self.latest: dict[int, UnitFrame] = {}
        self.serial = 0
        self.finished = False
        self.survey = Backdrop(dish.overview(), Placement(0.0, 0.0, OVERVIEW_UM_PER_PX), CHANNELS)
        self.log(LoopStarted(0.0, CHANNELS, tuple(r.name for r in ROUTINES)))
        for cell in dish.rng.choice(len(dish.cells.xy), units, replace=False):
            self.join(Unit.new(dish.outline_of(int(cell)), ""), int(cell), 0.0)

    def now_s(self) -> float:
        return time.monotonic() - self.started

    def log(self, event: LoopEvent) -> None:
        self.fold.take(event)
        self.told = None

    # What the panel reads.
    def snapshot(self) -> Snapshot:
        # The log is folded into a view only when it has grown, not on every frame.
        if self.told is None:
            self.told = (self.fold.view(), self.statuses())
        view, statuses = self.told
        return snapshot_of(view, statuses, self.now_s(), self.order)

    def statuses(self) -> tuple[RoutineStatus, ...]:
        return routine_statuses(self.fold)

    def backdrop(self) -> Backdrop:
        return self.survey

    def frames(self) -> Mapping[int, UnitFrame]:
        return self.latest

    def field_um(self) -> float:
        return FIELD_UM

    # The changes the panel sends.
    def pause(self, unit: int, paused: bool) -> None:
        self.inbox.append(partial(self.take_pause, unit, paused))

    def run_now(self, unit: int) -> None:
        self.inbox.append(partial(self.take_run_now, unit))

    def remove(self, unit: int) -> None:
        self.inbox.append(partial(self.take_remove, unit))

    def reorder(self, order: tuple[RoutineKey, ...]) -> None:
        self.order = order

    def add_unit(self, unit: Unit) -> None:
        self.inbox.append(partial(self.take_unit, unit))

    def present(self, unit: int) -> bool:
        fold = self.fold.units.get(unit)
        return fold is not None and not fold.left

    def take_pause(self, unit: int, paused: bool, now_s: float) -> None:
        if self.present(unit) and self.fold.units[unit].paused != paused:
            self.log(UnitPaused(now_s, unit) if paused else UnitResumed(now_s, unit))

    def take_run_now(self, unit: int, now_s: float) -> None:
        if self.present(unit):
            self.asked[unit] = self.asked.get(unit, 0) + 1
            self.log(RunNowAsked(now_s, unit, self.asked[unit]))

    def take_remove(self, unit: int, now_s: float) -> None:
        if self.present(unit):
            self.log(UnitLeft(now_s, unit))

    def take_unit(self, unit: Unit, now_s: float) -> None:
        if not fits(unit.outline, FIELD_UM):
            self.log(UnitRefused(now_s, unit.key, "larger than one field of view"))
            return
        x, y = middle(unit.outline)
        cell = self.dish.nearest(x, y)
        cx, cy = self.dish.at(cell, now_s)
        near = float(np.hypot(cx - x, cy - y)) <= FOLLOW_UM
        self.join(unit, cell if near else None, now_s)
        self.finished = False

    def join(self, unit: Unit, cell: int | None, now_s: float) -> None:
        number = len(self.fold.units) + 1
        self.cells[number] = cell
        self.log(UnitJoined(now_s, number, unit.key, unit.group, unit.outline))
        for index, routine in enumerate(ROUTINES):
            self.scans[(number, routine.name)] = 0
            due = now_s + 1.5 * number + 4.0 * index
            self.log(StepDue(now_s, number, routine.name, due))

    # The loop itself, stepped by the demo's clock.
    def step(self) -> None:
        now_s = self.now_s()
        busy = self.busy
        if busy is not None and now_s >= busy.ends_s:
            self.end_phase(busy, now_s)
        if self.busy is None:
            self.between_steps(now_s)

    def between_steps(self, now_s: float) -> None:
        inbox, self.inbox = self.inbox, []
        for change in inbox:
            change(now_s)
        statuses = self.statuses()
        if not statuses and not self.finished:
            self.finished = True
            self.log(LoopFinished(now_s))
            return
        chosen = next_to_run(statuses, now_s, self.order)
        if chosen is not None:
            self.begin(chosen, now_s)

    def where(self, unit: int, now_s: float) -> tuple[float, float]:
        """Where the unit truly is now: on its cell, or still where it was last tracked."""
        cell = self.cells[unit]
        return self.fold.units[unit].tracked if cell is None else self.dish.at(cell, now_s)

    def begin(self, status: RoutineStatus, now_s: float) -> None:
        x, y = self.where(status.unit, now_s)
        here = self.busy_at()
        moving_s = 0.4 + float(np.hypot(x - here[0], y - here[1])) / 1500.0
        self.busy = Busy(status.key, Phase.MOVING, now_s + min(moving_s, 2.5), status.due_s, x, y)
        self.log(MovingToUnit(now_s, status.unit, status.routine, x, y))

    def busy_at(self) -> tuple[float, float]:
        moves = [e for e in self.fold.recent if isinstance(e, MovingToUnit)]
        return (moves[-1].x_um, moves[-1].y_um) if moves else (0.0, 0.0)

    def end_phase(self, busy: Busy, now_s: float) -> None:
        unit, name = busy.key
        (routine,) = [r for r in ROUTINES if r.name == name]
        if busy.phase is Phase.MOVING:
            focus = self.scans[busy.key] % routine.focus_every == 0 or self.fold.units[unit].lost
            if focus:
                busy.phase, busy.ends_s = Phase.FOCUSING, now_s + FOCUS_S
                self.log(FocusingAtUnit(now_s, unit, name))
                return
        elif busy.phase is Phase.FOCUSING:
            z = 100.0 + float(self.dish.rng.normal(0, 0.4))
            self.log(FocusChecked(now_s, unit, name, z, z - 100.0))
        if busy.phase is not Phase.SCANNING:
            busy.phase, busy.ends_s = Phase.SCANNING, now_s + SCAN_S
            self.log(ScanningUnit(now_s, unit, name))
            return
        self.finish_scan(busy, routine, now_s)

    def finish_scan(self, busy: Busy, routine: FakeRoutine, now_s: float) -> None:
        unit, name = busy.key
        x, y = self.where(unit, now_s)
        found = float(self.dish.rng.random()) > LOST_SHARE
        planes = self.dish.field(x, y, now_s)
        self.serial += 1
        tag = FrameTag(unit, name, x - FIELD_UM / 2, y - FIELD_UM / 2)
        self.latest[unit] = UnitFrame(planes, tag, FIELD_UM / FIELD_PX, self.serial)
        frame = self.scans[busy.key]
        shift = (x - busy.x_um, y - busy.y_um)
        late = max(0.0, now_s - busy.due_s)
        self.log(UnitScanned(now_s, unit, name, frame, late, x, y, 100.0, shift, found, 20.0, ()))
        means = tuple(float(plane.mean()) for plane in planes)
        self.log(Measured(now_s, unit, name, "mean", means))
        self.scans[busy.key] = frame + 1
        if frame + 1 >= routine.scans:
            self.log(RoutineEnded(now_s, unit, name))
        else:
            self.log(StepDue(now_s, unit, name, now_s + routine.every_s))
        self.busy = None
