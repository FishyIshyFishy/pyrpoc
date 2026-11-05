"""The orchestrator: runs every unit's routines, the earliest due step first, tracking each unit
and keeping focus before it is scanned, and logging everything it does."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyrpoc.core.program_input import ParameterError
from pyrpoc.plugins.program_io.outputs import FrameTag, LoopEvent, Unit

from ....focus.tenengrad import tenengrad
from ...analysis.content import content_channels, content_region
from ...interfaces.field import FieldOfView
from ...interfaces.microscope import Microscope
from ...interfaces.progress import Progress
from .focus_keeper import FocusKeeper
from .queue import Queue, Thread
from .routine import Record, RoutineSpec, Scan, Seen, Step, Wait
from .sink import LoopSink
from .steering import Steering
from .survey_map import SurveyMap
from .tracking import Track, Tracker, crop_centred, outline_mask, start_track, window_px

# The longest the loop waits at once, so edits and stops are taken up promptly.
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


def check_units(units: Sequence[Unit], field: FieldOfView) -> None:
    """Refuse units the loop cannot run: none at all, or one larger than a field."""
    if not units:
        raise ParameterError("needs at least one unit: draw them over the survey in Autonomy")
    for unit in units:
        height, width = window_px(unit, field.um_per_px)
        if max(height, width) > field.pixels:
            raise ParameterError(f"unit {unit.id} is larger than one field of view")


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
        self.field = setup.field
        self.scan = setup.field.scan()
        self.tracker = Tracker(setup.field, setup.survey.motion)
        self.focus = FocusKeeper(setup.survey.focus, setup.focus_check_s, setup.final_step_um)
        self.queue = Queue()
        self.tracks: dict[int, Track] = {}
        self.frames: dict[str, int] = {}
        self.lost: set[int] = set()
        # Every unit run on, as drawn when it started, removed ones too: their frames remain.
        self.ran: dict[int, Unit] = {}
        self.paused: frozenset[int] = frozenset()

    def add_unit(self, unit: Unit, now_s: float) -> None:
        """Start tracking ``unit`` from the survey, and start each routine for it."""
        seen = self.setup.survey.seen_at(unit)
        self.tracks[unit.id] = start_track(unit, self.setup.survey.overview, self.field, seen)
        x, y = unit.centre
        outline = [list(point) for point in unit.outline]
        details = {"group": unit.group, "outline": outline, "x_um": x, "y_um": y}
        self.log_event("added", unit.id, "", details)
        self.ran[unit.id] = unit
        self.sink.keep_units(tuple(self.ran.values()))
        for spec in self.setup.routines:
            thread = Thread(unit.id, spec.name, spec.start(unit), now_s)
            self.queue.add(thread)
            self.log("due", thread, due_s=now_s)

    def log(self, event: str, thread: Thread | None, **details: Any) -> None:
        unit = None if thread is None else thread.unit
        routine = "" if thread is None else thread.routine
        self.log_event(event, unit, routine, details)

    def log_event(
        self, event: str, unit: int | None, routine: str, details: dict[str, Any]
    ) -> None:
        self.sink.log(LoopEvent(self.scope.now(), event, unit, routine, details))

    def run(self, units: Sequence[Unit]) -> None:
        """Run every thread of ``units``, and of units added since, to its end, or until the
        run is stopped."""
        routines = [r.name for r in self.setup.routines]
        self.log("start", None, channels=list(self.scope.channel_labels), routines=routines)
        now = self.scope.now()
        for unit in units:
            self.add_unit(unit, now)
        while self.queue.active():
            self.progress.check_cancel()
            self.between_steps()
            runnable = self.queue.runnable()
            if not runnable:
                self.scope.wait(IDLE_S)
                continue
            thread, now = runnable[0], self.scope.now()
            if thread.due_s > now:
                self.scope.wait(min(thread.due_s - now, IDLE_S))
                continue
            self.advance(thread)
        self.log("finish", None)

    def between_steps(self) -> None:
        """Take up what the panel asked for since the last step."""
        self.steer_units()
        self.steer_pauses()
        self.steer_run_now()

    def steer_units(self) -> None:
        """Start units drawn since the loop began, and stop units deleted."""
        units = {unit.id: unit for unit in self.steering.units()}
        now = self.scope.now()
        for uid, unit in units.items():
            if uid not in self.tracks:
                self.add_unit(unit, now)
        removed = [uid for uid in self.tracks if uid not in units and self.queue.of_unit(uid)]
        for uid in removed:
            for thread in self.queue.of_unit(uid):
                thread.finished = True
            self.log_event("removed", uid, "", {})

    def steer_pauses(self) -> None:
        paused = self.steering.paused()
        if paused == self.paused:
            return
        for uid in paused ^ self.paused:
            self.log_event("paused" if uid in paused else "resumed", uid, "", {})
        self.paused = paused
        for thread in self.queue.active():
            thread.paused = thread.unit in paused

    def steer_run_now(self) -> None:
        for uid in self.steering.run_now():
            threads = self.queue.of_unit(uid)
            for thread in threads:
                thread.urgent = True
            # Logged only for a unit it can run, so the log never shows a request it ignored.
            if threads:
                self.log_event("run now", uid, "", {})

    def next_step(self, thread: Thread) -> Step | None:
        """The thread's next step, or None when its routine has ended or failed."""
        try:
            step = thread.steps.send(thread.sending) if thread.started else next(thread.steps)
        except StopIteration:
            thread.finished = True
            self.log("done", thread)
            return None
        except (ValueError, ArithmeticError, IndexError, KeyError) as exc:
            # A routine's own mistake ends only that routine, and is logged.
            thread.finished = True
            self.log("error", thread, message=str(exc))
            return None
        thread.started, thread.sending = True, None
        return step

    def advance(self, thread: Thread) -> None:
        """Run the thread's next step whole."""
        step = self.next_step(thread)
        if isinstance(step, Wait):
            thread.due_s = self.scope.now() + step.seconds
            self.log("due", thread, due_s=thread.due_s)
        elif isinstance(step, Record):
            self.log("record", thread, name=step.name, values=list(step.values))
        elif isinstance(step, Scan):
            thread.sending = self.scan_unit(thread, step)

    def keep_focus(self, thread: Thread, x_um: float, y_um: float) -> None:
        """Find focus again on this unit if a check is due."""
        now = self.scope.now()
        if not self.focus.due(now):
            return
        self.scope.move_z(self.focus.z_at(x_um, y_um))
        frame = self.field.on_dish(self.scope.scan(self.scan, ()))
        channels = content_channels(frame)
        if not channels:
            return
        weights = content_region(frame, channels)

        def measure(z_um: float) -> float:
            self.scope.move_z(z_um)
            return tenengrad(
                self.field.on_dish(self.scope.scan(self.scan, ()))[list(channels)], weights
            )

        z = self.focus.check(measure, x_um, y_um, now)
        self.log("focus", thread, z_um=z, offset_um=self.focus.offset_um)

    def scan_unit(self, thread: Thread, step: Scan) -> Seen:
        """Scan a field centred where the unit was last seen, find it there, and give the
        routine its image."""
        track = self.tracks[thread.unit]
        due = thread.due_s
        x, y = track.centre
        self.scope.move_xy(x, y)
        self.keep_focus(thread, x, y)
        z = self.focus.z_at(x, y)
        self.scope.move_z(z)
        frame = self.field.on_dish(self.scope.scan(self.scan, step.lines))
        now = self.scope.now()
        thread.urgent = False
        shift = self.find_unit(thread, track, frame, now)
        seen = self.unit_image(track, frame, shift, now)
        index = self.save_frame(thread, track, seen)
        cx, cy = track.centre
        self.log(
            "scan",
            thread,
            due_s=due,
            ran_s=now,
            late_s=max(0.0, now - due),
            frame=index,
            x_um=cx,
            y_um=cy,
            z_um=z,
            shift_um=list(shift),
            found=thread.unit not in self.lost,
            lines=[[line.port, line.line] for line in step.lines],
        )
        return seen

    def find_unit(
        self, thread: Thread, track: Track, frame: np.ndarray, now: float
    ) -> tuple[float, float]:
        """Follow the unit to where it is in ``frame``, returning how far it moved; a unit not
        found stays where it was and is logged as lost."""
        found = self.tracker.locate(track, frame, now)
        if found is None:
            x, y = track.centre
            radius = self.tracker.radius_um(now - track.seen_s)
            self.log("lost", thread, x_um=x, y_um=y, radius_um=radius)
            self.lost.add(thread.unit)
            return (0.0, 0.0)
        self.tracker.follow(track, found, frame, now)
        self.lost.discard(thread.unit)
        return found.shift_um

    def unit_image(
        self, track: Track, frame: np.ndarray, shift: tuple[float, float], now: float
    ) -> Seen:
        """The unit's window of the frame, centred on where it was found."""
        size = window_px(track.unit, self.field.um_per_px)
        centre = self.field.pixels / 2.0
        at = (centre + shift[0] / self.field.um_per_px, centre + shift[1] / self.field.um_per_px)
        return Seen(
            crop_centred(frame, at, size), outline_mask(track.unit, size, self.field.um_per_px), now
        )

    def save_frame(self, thread: Thread, track: Track, seen: Seen) -> int:
        """Add the unit's image, before the log row that names it."""
        index = self.frames.get(thread.key, 0)
        self.frames[thread.key] = index + 1
        cx, cy = track.centre
        height, width = seen.mask.shape
        left = cx - width * self.field.um_per_px / 2.0
        top = cy - height * self.field.um_per_px / 2.0
        tag = FrameTag(thread.unit, thread.routine, left, top)
        self.sink.add_frame(seen.image, self.scope.channel_labels, tag)
        return index
