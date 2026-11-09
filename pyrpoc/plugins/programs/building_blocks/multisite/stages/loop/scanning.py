"""``UnitScanner``: one scan of one unit for its routine, with focus kept, the unit found again
and its frame kept."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pyrpoc.plugins.program_io.outputs import (
    FocusChecked,
    FocusingAtUnit,
    FocusingBeside,
    FocusNotFound,
    FrameTag,
    MovingToUnit,
    ScanningUnit,
    UnitScanned,
)

from ....focus.tenengrad import tenengrad
from ...analysis.content import content_channels, content_region
from ...interfaces.field import FieldOfView
from ...interfaces.microscope import Microscope
from ...interfaces.overview import Overview
from .focus_keeper import FocusKeeper
from .focus_site import Bounds, focus_site
from .queue import Thread
from .routine import Scan, Seen
from .sink import LoopSink
from .tracking import Track, Tracker, crop_centred, found_centre, outline_mask, window_px


class UnitScanner:
    """Scans units where each was last seen, and says how many frames each routine has taken of
    each unit."""

    def __init__(
        self,
        scope: Microscope,
        field: FieldOfView,
        tracker: Tracker,
        focus: FocusKeeper,
        overview: Overview,
        sink: LoopSink,
    ):
        self.scope = scope
        self.field = field
        self.check_geometry = field.scan()
        self.tracker = tracker
        self.focus = focus
        self.overview = overview
        self.sink = sink
        # How many frames each thread has taken, by its key.
        self.frames: dict[str, int] = {}

    def scan(self, thread: Thread, track: Track, step: Scan, avoid: Sequence[Bounds]) -> Seen:
        """Scan around where the unit was last seen, after a focus check clear of ``avoid`` if
        one is due, find the unit there, and give the routine its image."""
        due = thread.due_s
        x, y = track.centre
        self.keep_focus(thread, (x, y), avoid)
        self.sink.log(MovingToUnit(self.scope.now(), thread.unit, thread.routine, x, y))
        self.scope.move_xy(x, y)
        z = self.focus.z_at(x, y, self.scope.now())
        self.sink.log(ScanningUnit(self.scope.now(), thread.unit, thread.routine))
        self.scope.move_z(z)
        around = self.field.around(self.tracker.side_px(track, self.scope.now()))
        frame = around.on_dish(self.scope.scan(around.scan(), step.lines))
        now = self.scope.now()
        radius = self.tracker.radius_um(now - track.seen_s)
        found = self.tracker.locate(track, frame, now)
        if found is not None:
            self.tracker.follow(track, found, frame, now)
        shift = (0.0, 0.0) if found is None else found.shift_um
        seen = self.unit_image(track, frame, shift, now, found is not None)
        index = self.save_frame(thread, track, seen)
        x_now, y_now = track.centre
        self.sink.log(
            UnitScanned(
                time_s=now,
                unit=thread.unit,
                routine=thread.routine,
                frame=index,
                late_s=max(0.0, now - due),
                x_um=x_now,
                y_um=y_now,
                z_um=z,
                shift_um=shift,
                found=found is not None,
                radius_um=radius,
                lines=tuple((line.port, line.line) for line in step.lines),
            )
        )
        return seen

    def check_site(
        self, thread: Thread, near: tuple[float, float], avoid: Sequence[Bounds]
    ) -> tuple[float, float]:
        """Where to check focus for ``thread``'s unit: beside it, or on it if nowhere near is
        clear of every unit and holds content."""
        now = self.scope.now()
        site = focus_site(near, avoid, self.overview, self.field.field_um)
        if site is None:
            self.sink.log(FocusingAtUnit(now, thread.unit, thread.routine))
            return near
        self.sink.log(FocusingBeside(now, thread.unit, thread.routine, *site))
        return site

    def keep_focus(
        self, thread: Thread, near: tuple[float, float], avoid: Sequence[Bounds]
    ) -> None:
        """Find focus again near the unit if a check is due."""
        if not self.focus.due(self.scope.now()):
            return
        x, y = self.check_site(thread, near, avoid)
        self.scope.move_xy(x, y)
        self.scope.move_z(self.focus.z_at(x, y, self.scope.now()))
        frame = self.field.on_dish(self.scope.scan(self.check_geometry, ()))
        channels = content_channels(frame)
        if not channels:
            return
        weights = content_region(frame, channels)

        def measure(z_um: float) -> float:
            self.scope.move_z(z_um)
            image = self.field.on_dish(self.scope.scan(self.check_geometry, ()))
            return tenengrad(image[list(channels)], weights)

        found = self.focus.find(measure, x, y, self.scope.now())
        now = self.scope.now()
        if found.at_edge:
            # Not a measure of focus but a bound on it.
            self.focus.missed(now)
            self.sink.log(FocusNotFound(now, thread.unit, thread.routine))
            return
        offset = found.best_z_um - self.focus.plane.z_at(x, y)
        self.sink.log(FocusChecked(now, thread.unit, thread.routine, found.best_z_um, offset))
        self.focus.take(now, offset)

    def unit_image(
        self,
        track: Track,
        frame: np.ndarray,
        shift: tuple[float, float],
        now: float,
        found: bool,
    ) -> Seen:
        """The unit's window of the frame, centred on where it was found."""
        size = window_px(track.unit, self.field.um_per_px)
        at = found_centre(frame, shift, self.field.um_per_px)
        mask = outline_mask(track.unit, size, self.field.um_per_px)
        return Seen(crop_centred(frame, at, size), mask, now, found)

    def save_frame(self, thread: Thread, track: Track, seen: Seen) -> int:
        """Add the unit's image, before the event that names it."""
        index = self.frames.get(thread.key, 0)
        self.frames[thread.key] = index + 1
        cx, cy = track.centre
        height, width = seen.mask.shape
        left = cx - width * self.field.um_per_px / 2.0
        top = cy - height * self.field.um_per_px / 2.0
        tag = FrameTag(thread.unit, thread.routine, left, top)
        self.sink.add_frame(seen.image, self.scope.channel_labels, tag)
        return index
