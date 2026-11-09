"""Galvo calibration: the stage moves the sample by known distances, each frame is registered
whole against the frame before the move, and the fit of image shift against stage move gives
µm per volt."""

# Each move goes out and comes back, and the two shifts are differenced, so anything that drifts
# steadily between frames (the cells, the stage, focus) cancels out of the measurement.
#
# Every stop is approached from the same side, from a little short of it on the negative side.
# A stage's screw has play, so where it stops depends on which way it last moved; approached
# alike, every stop takes up the play alike, and it drops out of the measurement. Approached as
# they come, the play shortened or lengthened each shift by whether the stage had last moved out
# or back, which no line through the shifts can tell from the scale.
#
# The moves grow geometrically, and each is predicted from the scale the moves before it measured,
# so the shift is only ever searched for near where it should be. That keeps a regular sample (a
# lattice of beads, whose correlation peaks again one spacing away) from being read a spacing off:
# no search reaches further from its prediction than half the distance at which the view first
# matches itself again.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyrpoc.plugins.program_io.inputs import ScanGeometry
from pyrpoc.plugins.program_io.outputs import (
    AxisMeasured,
    AxisMeasuring,
    CalibrationPoint,
    FeatureFound,
    GalvoCalibration,
    StageMoved,
)

from ...analysis.view_shift import ambiguity_px, structure, view_shift
from ...interfaces.microscope import Microscope
from .fit import CalibrationError, fit
from .sink import CalibrationSink

Axis = Literal["x", "y"]

FIRST_MOVE_UM = 1.0
GROWTH = 4.0
MAX_TRIES = 12
# A shift this large is measured to a few percent, which is enough to size the real moves.
ROUGH_SHIFT_PX = 8.0
# How far from its predicted place the shift is searched for: a fixed allowance for the sample and
# stage drifting, plus a share of the predicted shift for the error of the scale that predicted it.
SEARCH_PX = 8.0
ROUGH_SEARCH_FRACTION = 0.25
SEARCH_FRACTION = 0.1
# The largest shift the view is allowed, as a share of the field along the axis, so the frames
# before and after a move always share most of what they show.
ROOM_FRACTION = 0.5
# The real moves, as fractions of that room.
MOVE_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
# How far short of each stop the stage turns to approach it: more than a stage's play.
APPROACH_UM = 10.0


def volts_per_pixel(scan: ScanGeometry, axis: Axis) -> float:
    if axis == "x":
        return 2.0 * scan.fast_axis_amplitude / scan.x_pixels
    return 2.0 * scan.slow_axis_amplitude / scan.y_pixels


def room_px(scan: ScanGeometry, axis: Axis) -> float:
    """The largest shift along ``axis`` the view is moved by."""
    return ROOM_FRACTION * (scan.x_pixels if axis == "x" else scan.y_pixels)


def slope_through_origin(moves: list[float], shifts: list[float]) -> float:
    m, s = np.asarray(moves), np.asarray(shifts)
    return float(np.sum(m * s) / np.sum(m * m))


@dataclass
class Calibrating:
    """One calibration in progress: the scan it takes, the channel it registers on, where the
    stage started, and the last frame taken there."""

    scope: Microscope
    scan: ScanGeometry
    sink: CalibrationSink
    channel: int
    origin: tuple[float, float]
    # The ``structure`` of the last frame taken at the origin.
    home: np.ndarray
    # The furthest from its prediction a shift is searched for, so it cannot be read a spacing
    # of a regular sample away from where it is.
    reach: float

    def take(self) -> np.ndarray:
        frame = self.scope.scan(self.scan, ())
        self.sink.show(frame)
        return structure(frame[self.channel])

    def shift_between(
        self, before: np.ndarray, after: np.ndarray, predicted: tuple[float, float], radius: float
    ) -> tuple[float, float]:
        found = view_shift(before, after, predicted, min(radius, self.reach))
        if found is None:
            raise CalibrationError(
                "the view after a move did not match the view before it: the move took it "
                "too far, or the sample changed or went out of focus"
            )
        return found[0]

    def shift_px(self, axis: Axis, move_um: float, predicted_px: float, radius: float) -> float:
        """How far the view moves along ``axis`` when the stage moves ``move_um``, where it is
        predicted to move ``predicted_px`` give or take ``radius``."""
        index = 0 if axis == "x" else 1
        self.go(axis, move_um)
        predicted = [0.0, 0.0]
        predicted[index] = predicted_px
        moved = self.take()
        out = self.shift_between(self.home, moved, (predicted[0], predicted[1]), radius)
        self.go(axis, 0.0)
        back = self.take()
        returned = self.shift_between(moved, back, (-out[0], -out[1]), radius)
        # The frame back at the origin is the one the next move is measured from.
        self.home = back
        return (out[index] - returned[index]) / 2.0

    def go(self, axis: Axis, offset_um: float) -> None:
        """Move to the origin offset ``offset_um`` along ``axis``, approaching from the
        negative side."""
        x, y = self.origin
        for short in (APPROACH_UM, 0.0):
            along = offset_um - short
            if axis == "x":
                self.scope.move_xy(x + along, y)
            else:
                self.scope.move_xy(x, y + along)

    def log_move(self, axis: Axis, move_um: float, shift_px: float, rough: bool) -> None:
        self.sink.log(StageMoved(self.scope.now(), axis, move_um, shift_px, rough))

    def rough_scale(self, axis: Axis) -> float:
        """Pixels of shift per µm of move, from moves that grow until the shift is clear."""
        move, shift = FIRST_MOVE_UM, None
        for _ in range(MAX_TRIES):
            if shift is None:
                # Nothing is known yet, so the first shift is looked for anywhere it could be.
                predicted, radius = 0.0, room_px(self.scan, axis)
            else:
                predicted = shift * GROWTH
                radius = SEARCH_PX + ROUGH_SEARCH_FRACTION * abs(predicted)
            shift = self.shift_px(axis, move, predicted, radius)
            self.log_move(axis, move, shift, True)
            if abs(shift) >= ROUGH_SHIFT_PX:
                return shift / move
            move *= GROWTH
        raise CalibrationError(
            f"the stage moved {move / GROWTH:g} µm in {axis} but the image did not"
        )


def measure_axis(session: Calibrating, axis: Axis) -> tuple[float, float, list[CalibrationPoint]]:
    """One axis's µm per volt, its residual, and the points it was fitted to."""
    session.sink.log(AxisMeasuring(session.scope.now(), axis))
    scale = session.rough_scale(axis)
    room = room_px(session.scan, axis)
    moves: list[float] = []
    shifts: list[float] = []
    for fraction in MOVE_FRACTIONS:
        for sign in (1.0, -1.0):
            move = sign * fraction * room / abs(scale)
            predicted = scale * move
            radius = SEARCH_PX + SEARCH_FRACTION * abs(predicted)
            shift = session.shift_px(axis, move, predicted, radius)
            session.log_move(axis, move, shift, False)
            moves.append(move)
            shifts.append(shift)
            # Each move refines the scale that sizes and predicts the next, larger one.
            scale = slope_through_origin(moves, shifts)
    slope, residual = fit(np.asarray(moves), np.asarray(shifts))
    per_px = volts_per_pixel(session.scan, axis)
    # The stage carrying the sample one way moves its image the other.
    um_per_volt = -1.0 / (slope * per_px)
    points = [CalibrationPoint(axis, m, s * per_px) for m, s in zip(moves, shifts, strict=True)]
    session.sink.log(AxisMeasured(session.scope.now(), axis, um_per_volt, residual))
    return um_per_volt, residual, points


def start_calibrating(scope: Microscope, scan: ScanGeometry, sink: CalibrationSink) -> Calibrating:
    """Choose the channel to register on: the one whose structure two frames of the same place,
    taken one after the other, agree on best."""
    # The origin approached as every later stop at it will be, so the first frames match them.
    x, y = scope.xy()
    scope.move_xy(x - APPROACH_UM, y - APPROACH_UM)
    scope.move_xy(x, y)
    # Two frames of an unmoved sample differ only by noise and a little drift, so how well they
    # correlate is how well that channel's structure stands above its noise.
    first = scope.scan(scan, ())
    sink.show(first)
    second = scope.scan(scan, ())
    sink.show(second)
    planes = [structure(plane) for plane in second]
    agreement = []
    for a, b in zip(first, planes, strict=True):
        found = view_shift(structure(a), b, (0.0, 0.0), SEARCH_PX)
        agreement.append(-np.inf if found is None else found[1])
    channel = int(np.argmax(agreement))
    if not np.isfinite(agreement[channel]):
        raise CalibrationError(
            "nothing in view stands clear of the noise: move to where the sample is in focus "
            "and try again"
        )
    home = planes[channel]
    room = max(room_px(scan, "x"), room_px(scan, "y"))
    reach = 0.5 * ambiguity_px(home, room)
    sink.log(FeatureFound(scope.now(), scope.channel_labels[channel]))
    return Calibrating(scope, scan, sink, channel, (x, y), home, reach)


def calibrate(scope: Microscope, scan: ScanGeometry, sink: CalibrationSink) -> GalvoCalibration:
    """Measure both axes with ``scan``, showing every frame taken in ``sink``."""
    session = start_calibrating(scope, scan, sink)
    try:
        x = measure_axis(session, "x")
        y = measure_axis(session, "y")
    except BaseException:
        scope.stop()
        raise
    channel = scope.channel_labels[session.channel]
    return GalvoCalibration(x[0], y[0], x[1], y[1], channel, (*x[2], *y[2]))
