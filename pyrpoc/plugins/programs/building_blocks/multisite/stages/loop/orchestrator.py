"""The orchestrator: runs every unit's routines, the earliest due step first, takes the commands
that change its units between steps, and logs everything it does."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from pyrpoc.core.program_input import ParameterError
from pyrpoc.plugins.program_io.outputs import (
    CommandRefused,
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
    UnitResumed,
)
from pyrpoc.plugins.program_io.requests import (
    AddUnit,
    EditUnit,
    PauseUnit,
    RemoveUnit,
    ResumeUnit,
    RunUnitNow,
    UnitCommand,
    command_key,
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


def fits(unit: Unit, field: FieldOfView) -> bool:
    """Whether ``unit``, with its margin, fits in one field."""
    return max(window_px(unit, field.um_per_px)) <= field.pixels


def check_units(units: Sequence[Unit], field: FieldOfView) -> None:
    """Refuse starting units the loop cannot run: none at all, or one larger than a field."""
    if not units:
        raise ParameterError("needs at least one unit drawn over the survey")
    if not all(fits(unit, field) for unit in units):
        raise ParameterError("a unit is larger than one field of view")


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

    def start(self, units: Sequence[Unit]) -> None:
        """Start afresh on ``units``."""
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
            thread.paused = record.paused
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

    def run(self) -> None:
        """Run every routine to its end, taking commands between steps, or until stopped."""
        while self.queue.active():
            self.progress.check_cancel()
            self.take_commands()
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

    def take_commands(self) -> None:
        """Carry out the commands given since the last step, and keep the units if any
        changed."""
        changed = [self.take(command) for command in self.steering.commands()]
        if any(changed):
            self.sink.keep_units(self.roster.records())

    def take(self, command: UnitCommand) -> bool:
        """Carry out ``command``, returning whether it changed the units."""
        match command:
            case AddUnit(unit):
                return self.add(command, unit)
            case EditUnit(unit):
                return self.edit(command, unit)
            case RemoveUnit():
                return self.remove(command)
            case PauseUnit() | ResumeUnit():
                return self.pause(command, isinstance(command, PauseUnit))
            case RunUnitNow():
                self.run_now(command)
                return False

    def refuse(self, command: UnitCommand, reason: str) -> None:
        self.sink.log(CommandRefused(self.scope.now(), command.word, command_key(command), reason))

    def present(self, command: UnitCommand) -> UnitRecord | None:
        """The unit ``command`` names, or None, refusing it, when the loop does not have it."""
        record = self.roster.present(command_key(command))
        if record is None:
            self.refuse(command, "the loop has no such unit")
        return record

    def add(self, command: AddUnit, unit: Unit) -> bool:
        if self.roster.has(unit.key):
            self.refuse(command, "a unit with this key already joined")
            return False
        if not fits(unit, self.setup.field):
            self.refuse(command, "larger than one field of view")
            return False
        self.join(unit, self.scope.now())
        return True

    def edit(self, command: EditUnit, unit: Unit) -> bool:
        record = self.present(command)
        if record is None:
            return False
        if not fits(unit, self.setup.field):
            self.refuse(command, "larger than one field of view")
            return False
        now = self.scope.now()
        if unit.outline != record.unit.outline:
            # A new outline says where the unit is now, so tracking starts again from there.
            self.tracks[record.number] = placed_track(unit, now)
        self.roster.update(replace(record, unit=unit))
        self.sink.log(UnitEdited(now, record.number, unit.group, unit.outline))
        return True

    def remove(self, command: RemoveUnit) -> bool:
        record = self.present(command)
        if record is None:
            return False
        for thread in self.queue.of_unit(record.number):
            thread.finished = True
        now = self.scope.now()
        self.roster.update(replace(record, left_s=now))
        self.sink.log(UnitLeft(now, record.number))
        return True

    def pause(self, command: PauseUnit | ResumeUnit, paused: bool) -> bool:
        record = self.present(command)
        if record is None or record.paused == paused:
            return False
        for thread in self.queue.of_unit(record.number):
            thread.paused = paused
        self.roster.update(replace(record, paused=paused))
        event = UnitPaused if paused else UnitResumed
        self.sink.log(event(self.scope.now(), record.number))
        return True

    def run_now(self, command: RunUnitNow) -> None:
        record = self.present(command)
        if record is None:
            return
        threads = self.queue.of_unit(record.number)
        if not threads:
            self.refuse(command, "its routines have all ended")
            return
        for thread in threads:
            thread.urgent = True
        self.sink.log(RunNowAsked(self.scope.now(), record.number))

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
