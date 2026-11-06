"""``UnitScanner``: one scan of one unit for its routine, with focus kept, the unit found again
and its frame kept."""

from __future__ import annotations

import numpy as np

from pyrpoc.plugins.program_io.outputs import (
    FocusChecked,
    FocusingAtUnit,
    FrameTag,
    MovingToUnit,
    ScanningUnit,
    UnitScanned,
)

from ....focus.tenengrad import tenengrad
from ...analysis.content import content_channels, content_region
from ...interfaces.field import FieldOfView
from ...interfaces.microscope import Microscope
from .focus_keeper import FocusKeeper
from .queue import Thread
from .routine import Scan, Seen
from .sink import LoopSink
from .tracking import Track, Tracker, crop_centred, outline_mask, window_px


class UnitScanner:
    """Scans units where each was last seen, and says how many frames each routine has taken of
    each unit."""

    def __init__(
        self,
        scope: Microscope,
        field: FieldOfView,
        tracker: Tracker,
        focus: FocusKeeper,
        sink: LoopSink,
    ):
        self.scope = scope
        self.field = field
        self.scan_geometry = field.scan()
        self.tracker = tracker
        self.focus = focus
        self.sink = sink
        # How many frames each thread has taken, by its key.
        self.frames: dict[str, int] = {}

    def scan(self, thread: Thread, track: Track, step: Scan) -> Seen:
        """Scan a field centred where the unit was last seen, find it there, and give the
        routine its image."""
        due = thread.due_s
        x, y = track.centre
        self.sink.log(MovingToUnit(self.scope.now(), thread.unit, thread.routine, x, y))
        self.scope.move_xy(x, y)
        self.keep_focus(thread, x, y)
        z = self.focus.z_at(x, y)
        self.sink.log(ScanningUnit(self.scope.now(), thread.unit, thread.routine))
        self.scope.move_z(z)
        frame = self.field.on_dish(self.scope.scan(self.scan_geometry, step.lines))
        now = self.scope.now()
        radius = self.tracker.radius_um(now - track.seen_s)
        found = self.tracker.locate(track, frame, now)
        if found is not None:
            self.tracker.follow(track, found, frame, now)
        shift = (0.0, 0.0) if found is None else found.shift_um
        seen = self.unit_image(track, frame, shift, now)
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

    def keep_focus(self, thread: Thread, x_um: float, y_um: float) -> None:
        """Find focus again on this unit if a check is due."""
        now = self.scope.now()
        if not self.focus.due(now):
            return
        self.sink.log(FocusingAtUnit(now, thread.unit, thread.routine))
        self.scope.move_z(self.focus.z_at(x_um, y_um))
        frame = self.field.on_dish(self.scope.scan(self.scan_geometry, ()))
        channels = content_channels(frame)
        if not channels:
            return
        weights = content_region(frame, channels)

        def measure(z_um: float) -> float:
            self.scope.move_z(z_um)
            image = self.field.on_dish(self.scope.scan(self.scan_geometry, ()))
            return tenengrad(image[list(channels)], weights)

        z = self.focus.check(measure, x_um, y_um, now)
        self.sink.log(
            FocusChecked(self.scope.now(), thread.unit, thread.routine, z, self.focus.offset_um)
        )

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
