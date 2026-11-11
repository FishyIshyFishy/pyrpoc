"""``LoopFeed``: a survey and the newest loop run over it, as the panel reads them, and the
changes the panel sends that loop through the units it is given."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeVar

import numpy as np

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.recording import Recording
from pyrpoc.plugins.program_io.inputs import SURVEY, UNITS
from pyrpoc.plugins.program_io.outputs import (
    FrameTag,
    LoopEvent,
    LoopFold,
    LoopLogOutput,
    LoopView,
    Outline,
    Placement,
    SurveyOutput,
    Unit,
    UnitFramesOutput,
    UnitRefused,
    UnitsOutput,
)

from .state.clock import LoopClock
from .state.pending import Change, Pending
from .state.standing import Snapshot

OutputT = TypeVar("OutputT", bound=ProgramOutput)


@dataclass(frozen=True)
class Backdrop:
    """The survey's ``(C, H, W)`` overview, where it lies on the dish, and its channels."""

    planes: np.ndarray
    placement: Placement
    channels: tuple[str, ...]


@dataclass(frozen=True)
class UnitFrame:
    """A unit's latest ``(C, H, W)`` frame, where it lies, its pixel size in µm, and a count
    that rises with every new frame."""

    planes: np.ndarray
    tag: FrameTag
    um_per_px: float
    serial: int


def output_of(recording: Recording, kind: type[OutputT]) -> OutputT | None:
    """The recording's output of type ``kind``, found by its type, not its name."""
    return next((each for each in recording.outputs.values() if isinstance(each, kind)), None)


def reads_survey(recording: Recording, survey: SurveyOutput) -> bool:
    """Whether ``recording`` was given ``survey``, saved in the same folder, as its survey."""
    folder = survey.recording.folder
    if folder is None or recording.folder != folder:
        return False
    given = SURVEY.rule.encode(survey)
    return any(part.parameters.get(SURVEY.key) == given for part in recording.parts)


def loop_over(survey: SurveyOutput, library: Library) -> LoopLogOutput | None:
    """The log of the newest loop that read ``survey``."""
    for recording in reversed(library.recordings()):
        log = output_of(recording, LoopLogOutput)
        if log is not None and reads_survey(recording, survey):
            return log
    return None


class LoopFeed:
    def __init__(self, survey: SurveyOutput, library: Library, inputs: ProgramInputs):
        self.survey, self.library, self.inputs = survey, library, inputs
        self.pending = Pending()
        self.follow(None)

    def follow(self, log: LoopLogOutput | None) -> None:
        """Start reading ``log`` from its first event."""
        self.log = log
        self.fold = LoopFold()
        self.view: LoopView | None = None
        self.events_read = 0
        self.frames_read = 0
        self.frames: dict[int, UnitFrame] = {}
        self.refused: dict[str, str] = {}
        self.clock = LoopClock()

    def backdrop(self) -> Backdrop | None:
        """The overview, once the survey has finished."""
        if self.survey.finding is None:
            return None
        planes, placement = self.survey.overview()
        return Backdrop(planes, placement, self.survey.channels)

    def field_um(self) -> float:
        """The side of one scan field, in µm: one survey tile."""
        return self.survey.field_um

    def update(self, wall_s: float) -> Snapshot:
        """Take whatever the loop has logged and imaged since the last update."""
        log = loop_over(self.survey, self.library)
        if log is not self.log:
            self.follow(log)
        self.pending.settle(self.take_events(wall_s), wall_s)
        self.take_frames()
        return self.snapshot(wall_s)

    def take_events(self, wall_s: float) -> list[LoopEvent]:
        if self.log is None:
            return []
        new = self.log.since(self.events_read)
        if not new:
            return new
        self.events_read += len(new)
        for event in new:
            self.fold.take(event)
            if isinstance(event, UnitRefused):
                self.refused[event.key] = event.reason
        self.view = self.fold.view()
        self.clock.seen(new[-1].time_s, wall_s)
        return new

    def take_frames(self) -> None:
        frames = None if self.log is None else output_of(self.log.recording, UnitFramesOutput)
        if frames is None:
            return
        for planes, tag in frames.since(self.frames_read):
            self.frames_read += 1
            self.frames[tag.unit] = UnitFrame(planes, tag, self.survey.um_per_px, self.frames_read)

    def snapshot(self, wall_s: float) -> Snapshot:
        view = self.view
        running = self.log is not None and not self.log.recording.finished
        if running:
            now_s = self.clock.now(wall_s)
        else:
            now_s = 0.0 if view is None else view.started_s + view.elapsed_s
        joined = set() if view is None else {unit.key for unit in view.units}
        drawn = tuple(unit for unit in self.given() if unit.key not in joined)
        return Snapshot(view, now_s, running, drawn, dict(self.refused))

    # The changes the panel sends, all through the units the loop is given.
    def given(self) -> tuple[Unit, ...]:
        units: tuple[Unit, ...] = self.inputs.get(UNITS.key)
        return units

    def give(self, units: tuple[Unit, ...]) -> None:
        self.inputs.set(UNITS.key, units)

    def taken(self, number: int) -> Unit | None:
        """Unit ``number`` as the loop last took it, or None in the moment between its joining
        and the loop keeping its units."""
        units = None if self.log is None else output_of(self.log.recording, UnitsOutput)
        if units is None:
            return None
        return next((r.unit for r in units.records if r.number == number), None)

    def edit(self, number: int, change: Change, wall_s: float) -> None:
        """Give the loop unit ``number`` with ``change`` made to it."""
        taken = self.taken(number)
        if taken is None:
            return
        given = self.given()
        # A loop carried on is given no units at first, so the unit is given as it was taken.
        unit = next((each for each in given if each.key == taken.key), taken)
        if change is Change.RUN_NEXT:
            # The loop runs a unit now each time its count rises past the count it took.
            asked = max(unit.run_now_asked, taken.run_now_asked) + 1
            edited = replace(unit, run_now_asked=asked)
        else:
            edited = replace(unit, paused=change is Change.PAUSE)
        if any(each.key == taken.key for each in given):
            self.give(tuple(edited if each.key == taken.key else each for each in given))
        else:
            self.give((*given, edited))
        self.pending.send(number, change, wall_s)

    def remove(self, number: int, wall_s: float) -> None:
        """Stop giving the loop unit ``number``, which it then takes out."""
        taken = self.taken(number)
        if taken is None:
            return
        self.drop(taken.key)
        self.pending.send(number, Change.REMOVE, wall_s)

    def add(self, outline: Outline) -> None:
        # Groups are not drawn yet, so every unit takes the one plain group.
        self.give((*self.given(), Unit.new(outline, "")))

    def drop(self, key: str) -> None:
        self.give(tuple(unit for unit in self.given() if unit.key != key))
