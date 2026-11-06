"""The orchestrator: runs every unit's routines, the earliest due step first, takes the units
wanted between steps, and logs everything it does."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pyrpoc.plugins.program_io.outputs import (
    LoopFinished,
    LoopResumed,
    LoopStarted,
    Measured,
    RoutineEnded,
    RoutineFailed,
    RunNowAsked,
    StepDue,
    Unit,
    UnitEdited,
    UnitJoined,
    UnitLeft,
    UnitPaused,
    UnitRecord,
    UnitRefused,
    UnitResumed,
)

from ...interfaces.field import FieldOfView
from ...interfaces.microscope import Microscope
from ...interfaces.progress import Progress
from .focus_keeper import FocusKeeper
from .queue import Queue, Thread
from .resume import Resumption
from .roster import Roster
from .routine import NO_HISTORY, Record, RoutineSpec, Scan, Step, Wait
from .scanning import UnitScanner
from .sink import LoopSink
from .steering import Steering
from .survey_map import SurveyMap
from .tracking import Track, Tracker, placed_track, start_track, window_px

# The longest the loop waits at once, so commands and stops are taken up promptly.
IDLE_S = 1.0


@dataclass(frozen=True)
class LoopSetup:
    """What a loop runs from: the survey's map of the dish, the field each scan covers, how
    often focus is checked and to what step, and the routines."""

    survey: SurveyMap
    field: FieldOfView
    focus_check_s: float
    final_step_um: float
    routines: tuple[RoutineSpec, ...]


def fits(unit: Unit, um_per_px: float, pixels: int) -> bool:
    """Whether ``unit``, with its margin, fits in one field ``pixels`` a side at ``um_per_px``."""
    return max(window_px(unit, um_per_px)) <= pixels


class Orchestrator:
    """One loop in progress."""

    def __init__(
        self,
        progress: Progress,
        scope: Microscope,
        setup: LoopSetup,
        sink: LoopSink,
        steering: Steering,
    ):
        self.progress = progress
        self.scope = scope
        self.setup = setup
        self.sink = sink
        self.steering = steering
        tracker = Tracker(setup.field, setup.survey.motion)
        self.focus = FocusKeeper(setup.survey.focus, setup.focus_check_s, setup.final_step_um)
        self.scanner = UnitScanner(scope, setup.field, tracker, self.focus, sink)
        self.queue = Queue()
        self.roster = Roster(())
        self.tracks: dict[int, Track] = {}
        # The units last taken from the steering, which the next ones are compared with.
        self.taken: tuple[Unit, ...] = ()

    def start(self, units: tuple[Unit, ...]) -> None:
        """Start afresh on ``units``."""
        self.taken = units
        names = tuple(spec.name for spec in self.setup.routines)
        now = self.scope.now()
        self.sink.log(LoopStarted(now, tuple(self.scope.channel_labels), names))
        for unit in units:
            self.join(unit, now)
        self.sink.keep_units(self.roster.records())

    def resume(self, reached: Resumption) -> None:
        """Carry on from where a loop cut short had reached."""
        self.roster = Roster(reached.records.values())
        self.focus.offset_um = reached.focus_offset_um
        if reached.focus_checked_s is not None:
            self.focus.checked_s = reached.focus_checked_s
        self.sink.log(LoopResumed(self.scope.now()))
        for record in self.roster.records():
            if record.left_s is None:
                self.carry_on(record, reached)
        self.sink.keep_units(self.roster.records())

    def carry_on(self, record: UnitRecord, reached: Resumption) -> None:
        """Track ``record``'s unit from where it was last known to be, and carry on each of its
        routines that had not ended."""
        placed = reached.placed[record.number]
        if placed is None:
            self.tracks[record.number] = self.survey_track(record.unit)
        else:
            at = record.unit.moved_to(placed.x_um, placed.y_um)
            self.tracks[record.number] = placed_track(at, placed.seen_s)
        for spec in self.setup.routines:
            progress = reached.threads.get((record.number, spec.name))
            if progress is None or progress.ended:
                continue
            steps = spec.start(record.unit, progress.history())
            thread = Thread(record.number, spec.name, steps, progress.due_s)
            thread.paused = record.unit.paused
            self.queue.add(thread)
            self.scanner.frames[thread.key] = progress.frames

    def survey_track(self, unit: Unit) -> Track:
        """A track from the survey's overview, where the unit was drawn."""
        survey = self.setup.survey
        return start_track(unit, survey.overview, self.setup.field, survey.seen_at(unit))

    def join(self, unit: Unit, now_s: float) -> None:
        """Number ``unit``, start tracking it from the survey, and start each routine for it."""
        number = self.roster.join(unit, now_s).number
        self.tracks[number] = self.survey_track(unit)
        self.sink.log(UnitJoined(now_s, number, unit.key, unit.group, unit.outline))
        for spec in self.setup.routines:
            self.queue.add(Thread(number, spec.name, spec.start(unit, NO_HISTORY), now_s))
            self.sink.log(StepDue(now_s, number, spec.name, now_s))
        if unit.paused:
            self.pause(number, True)

    def run(self) -> None:
        """Run every routine to its end, taking the units wanted between steps, or until
        stopped."""
        while self.queue.active():
            self.progress.check_cancel()
            self.take_units()
            runnable = self.queue.runnable()
            if not runnable:
                self.scope.wait(IDLE_S)
                continue
            thread, now = runnable[0], self.scope.now()
            if thread.due_s > now:
                self.scope.wait(min(thread.due_s - now, IDLE_S))
                continue
            self.advance(thread)
        self.sink.log(LoopFinished(self.scope.now()))

    def take_units(self) -> None:
        """Bring the loop to the units wanted now, and keep the units if any changed."""
        wanted = self.steering.units()
        if wanted == self.taken:
            return
        before = {unit.key: unit for unit in self.taken}
        self.taken = wanted
        keys = {unit.key for unit in wanted}
        # Only a unit wanted before and not now leaves, so a set that never named a unit, such
        # as the empty one a carried-on loop starts with, takes none away.
        changed = [self.remove(key) for key in before.keys() - keys]
        changed += [self.change(unit) for unit in wanted if before.get(unit.key) != unit]
        if any(changed):
            self.sink.keep_units(self.roster.records())

    def fits(self, unit: Unit) -> bool:
        field = self.setup.field
        return fits(unit, field.um_per_px, field.pixels)

    def refuse(self, key: str, reason: str) -> None:
        self.sink.log(UnitRefused(self.scope.now(), key, reason))

    def change(self, unit: Unit) -> bool:
        """Join ``unit``, or bring the unit with its key to it, returning whether the units
        changed."""
        record = self.roster.present(unit.key)
        if record is not None:
            return self.edit(record, unit)
        if self.roster.has(unit.key):
            self.refuse(unit.key, "it was removed, and a unit drawn again is a new unit")
            return False
        if not self.fits(unit):
            self.refuse(unit.key, "larger than one field of view")
            return False
        self.join(unit, self.scope.now())
        return True

    def edit(self, record: UnitRecord, unit: Unit) -> bool:
        """Bring ``record``'s unit to ``unit``: its outline and group, whether it is paused, and
        whether it is asked to run now."""
        old = record.unit
        drawn = self.redraw(record, unit)
        new = replace(drawn, paused=unit.paused, run_now_asked=unit.run_now_asked)
        if new == old:
            return False
        self.roster.update(replace(record, unit=new))
        if new.paused != old.paused:
            self.pause(record.number, new.paused)
        if new.run_now_asked > old.run_now_asked:
            self.run_now(record.number, new.run_now_asked)
        return True

    def redraw(self, record: UnitRecord, unit: Unit) -> Unit:
        """``record``'s unit with ``unit``'s outline and group when they changed and fit, and as
        it was otherwise."""
        old = record.unit
        if (unit.outline, unit.group) == (old.outline, old.group):
            return old
        if not self.fits(unit):
            self.refuse(unit.key, "larger than one field of view")
            return old
        now = self.scope.now()
        if unit.outline != old.outline:
            # A new outline says where the unit is now, so tracking starts again from there.
            self.tracks[record.number] = placed_track(unit, now)
        self.sink.log(UnitEdited(now, record.number, unit.group, unit.outline))
        return replace(old, outline=unit.outline, group=unit.group)

    def remove(self, key: str) -> bool:
        """Stop running anything on the unit with ``key``, returning whether it was running."""
        record = self.roster.present(key)
        if record is None:
            return False
        for thread in self.queue.of_unit(record.number):
            thread.finished = True
        now = self.scope.now()
        self.roster.update(replace(record, left_s=now))
        self.sink.log(UnitLeft(now, record.number))
        return True

    def pause(self, number: int, paused: bool) -> None:
        for thread in self.queue.of_unit(number):
            thread.paused = paused
        event = UnitPaused if paused else UnitResumed
        self.sink.log(event(self.scope.now(), number))

    def run_now(self, number: int, asked: int) -> None:
        """Put the unit's routines ahead of everything due; a paused one goes first once
        resumed."""
        for thread in self.queue.of_unit(number):
            thread.urgent = True
        self.sink.log(RunNowAsked(self.scope.now(), number, asked))

    def next_step(self, thread: Thread) -> Step | None:
        """The thread's next step, or None when its routine has ended or failed."""
        try:
            step = thread.steps.send(thread.sending) if thread.started else next(thread.steps)
        except StopIteration:
            thread.finished = True
            self.sink.log(RoutineEnded(self.scope.now(), thread.unit, thread.routine))
            return None
        except (ValueError, ArithmeticError, IndexError, KeyError) as exc:
            # A routine's own mistake ends only that routine, and is logged.
            thread.finished = True
            now = self.scope.now()
            self.sink.log(RoutineFailed(now, thread.unit, thread.routine, str(exc)))
            return None
        thread.started, thread.sending = True, None
        return step

    def advance(self, thread: Thread) -> None:
        """Run the thread's next step whole."""
        step = self.next_step(thread)
        if isinstance(step, Wait):
            thread.due_s = self.scope.now() + step.seconds
            self.sink.log(StepDue(self.scope.now(), thread.unit, thread.routine, thread.due_s))
        elif isinstance(step, Record):
            now = self.scope.now()
            self.sink.log(Measured(now, thread.unit, thread.routine, step.name, step.values))
        elif isinstance(step, Scan):
            thread.urgent = False
            thread.sending = self.scanner.scan(thread, self.tracks[thread.unit], step)
