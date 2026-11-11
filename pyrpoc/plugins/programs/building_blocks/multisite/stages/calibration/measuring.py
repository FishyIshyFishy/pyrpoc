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
#
# Each move is measured by where the stage reports it went, not where it was sent: a stage with
# encoders then lands wherever it likes without that showing as scatter, and one without reports
# its target, which is no worse.
#
# The moves are repeated in rounds until the scale is known well enough that its error moves the
# field's edge by under half a pixel at two standard errors; a scale known better than the image
# can show is not needed, and one known worse shows as seams and misplaced units. A stage that
# lands less repeatably takes more rounds to get there (the error falls as one over the root of
# the moves); one that cannot get there in MAX_ROUNDS still passes if its scale is known to
# WORST_UNCERTAINTY.
#
# The shift across the axis is measured too. The scan is taken to run along the stage's axes, and
# the angle between them says how far that holds: tan(angle) times half the field is how many
# pixels the field's edge is misplaced by.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyrpoc.plugins.program_io.inputs import ScanGeometry
from pyrpoc.plugins.program_io.outputs import (
    AxisMeasured,
    AxisMeasuring,
    AxisPrecision,
    CalibrationPoint,
    FeatureFound,
    GalvoCalibration,
    StageMoved,
)

from ...analysis.view_shift import ambiguity_px, structure, view_shift
from ...interfaces.microscope import Microscope
from .fit import CalibrationError, Line, cross_slope, fit
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
# How far, in pixels at the field's edge, the scale's error may move it at COVERAGE standard
# errors, before more rounds of moves are taken, up to MAX_ROUNDS.
EDGE_PX = 0.5
COVERAGE = 2.0
MAX_ROUNDS = 4
# A slope known no better than this fraction of itself did not measure the scale.
WORST_UNCERTAINTY = 0.01


def volts_per_pixel(scan: ScanGeometry, axis: Axis) -> float:
    if axis == "x":
        return 2.0 * scan.fast_axis_amplitude / scan.x_pixels
    return 2.0 * scan.slow_axis_amplitude / scan.y_pixels


def pixels_along(scan: ScanGeometry, axis: Axis) -> int:
    return scan.x_pixels if axis == "x" else scan.y_pixels


def room_px(scan: ScanGeometry, axis: Axis) -> float:
    """The largest shift along ``axis`` the view is moved by."""
    return ROOM_FRACTION * pixels_along(scan, axis)


def target_uncertainty(scan: ScanGeometry, axis: Axis) -> float:
    """The scale's standard error, as a fraction of it, at which the field's edge along ``axis``
    is known to ``EDGE_PX`` at ``COVERAGE`` standard errors."""
    return EDGE_PX / (COVERAGE * pixels_along(scan, axis) / 2.0)


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

    def shift_px(
        self, axis: Axis, move_um: float, predicted_px: float, radius: float
    ) -> tuple[float, float, float, float]:
        """Sent ``move_um`` along ``axis`` and back, how far the stage reports it moved along and
        across ``axis``, and how far the view moved along and across it, where it is predicted
        to move ``predicted_px`` along it give or take ``radius``."""
        index = 0 if axis == "x" else 1
        start = self.scope.xy()
        self.go(axis, move_um)
        there = self.scope.xy()
        predicted = [0.0, 0.0]
        predicted[index] = predicted_px
        moved = self.take()
        out = self.shift_between(self.home, moved, (predicted[0], predicted[1]), radius)
        self.go(axis, 0.0)
        back_at = self.scope.xy()
        back = self.take()
        returned = self.shift_between(moved, back, (-out[0], -out[1]), radius)
        # The frame back at the origin is the one the next move is measured from.
        self.home = back
        # Measured from both ends of the move, as the shifts are.
        moved_um = [there[i] - (start[i] + back_at[i]) / 2.0 for i in (0, 1)]
        shifted = [(out[i] - returned[i]) / 2.0 for i in (0, 1)]
        return moved_um[index], moved_um[1 - index], shifted[index], shifted[1 - index]

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
            distance, _, shift, _ = self.shift_px(axis, move, predicted, radius)
            self.log_move(axis, distance, shift, True)
            if abs(shift) >= ROUGH_SHIFT_PX:
                return shift / distance
            move *= GROWTH
        raise CalibrationError(
            f"the stage moved {move / GROWTH:g} µm in {axis} but the image did not"
        )


@dataclass
class Moves:
    """One axis's moves so far: how far the stage reports it went along the axis and across it,
    in µm, and how far the view moved along the axis and across it, in pixels."""

    moves: list[float]
    sideways: list[float]
    shifts: list[float]
    crosses: list[float]


def measure_round(session: Calibrating, axis: Axis, scale: float, taken: Moves) -> float:
    """One round of moves out to the room, each sized and predicted from ``scale``, the pixels
    per µm measured so far; returns the scale with the round's moves."""
    room = room_px(session.scan, axis)
    for fraction in MOVE_FRACTIONS:
        for sign in (1.0, -1.0):
            move = sign * fraction * room / abs(scale)
            predicted = scale * move
            radius = SEARCH_PX + SEARCH_FRACTION * abs(predicted)
            distance, sideways, shift, across = session.shift_px(axis, move, predicted, radius)
            session.log_move(axis, distance, shift, False)
            taken.moves.append(distance)
            taken.sideways.append(sideways)
            taken.shifts.append(shift)
            taken.crosses.append(across)
            # Each move refines the scale that sizes and predicts the next, larger one.
            scale = slope_through_origin(taken.moves, taken.shifts)
    return scale


def precise_line(session: Calibrating, axis: Axis, scale: float) -> tuple[Line, Moves, int]:
    """The line through rounds of moves, taken until it knows the scale to the target or the
    rounds run out, the moves, and how many rounds were taken."""
    target = target_uncertainty(session.scan, axis)
    taken = Moves([], [], [], [])
    rounds = 0
    while True:
        rounds += 1
        scale = measure_round(session, axis, scale, taken)
        line = fit(np.asarray(taken.moves), np.asarray(taken.shifts))
        if line.uncertainty <= target or rounds == MAX_ROUNDS:
            break
    if line.uncertainty > WORST_UNCERTAINTY or line.kept.sum() < len(taken.moves) // 2:
        raise CalibrationError(
            f"the measured shifts scatter too much about a line (the scale is uncertain by "
            f"{line.uncertainty:.1%}): try another spot"
        )
    return line, taken, rounds


@dataclass(frozen=True)
class AxisFit:
    """One axis measured: its line, the moves it was fitted to, and how many rounds they took."""

    axis: Axis
    line: Line
    taken: Moves
    rounds: int

    def um_per_volt(self, scan: ScanGeometry) -> float:
        # The stage carrying the sample one way moves its image the other.
        return -1.0 / (self.line.slope * volts_per_pixel(scan, self.axis))

    def points(self, scan: ScanGeometry) -> list[CalibrationPoint]:
        per_px = volts_per_pixel(scan, self.axis)
        pairs = zip(self.taken.moves, self.taken.shifts, strict=True)
        return [CalibrationPoint(self.axis, move, shift * per_px) for move, shift in pairs]

    def off_axis(self, other: AxisFit) -> float:
        """The tangent of the angle the image moved off this axis by as the stage moved along
        it, against the slope's own direction, so it reads 0 whichever way the scan runs."""
        # What the stage reports it strayed across the axis moved the image too, at the other
        # axis's scale; it is the stage's landing, not the angle, so it is taken out.
        strayed = other.line.slope * np.asarray(self.taken.sideways)
        crosses = np.asarray(self.taken.crosses) - strayed
        moves = np.asarray(self.taken.moves)
        return cross_slope(moves, crosses, self.line.kept) / self.line.slope

    def precision(self, other: AxisFit, scan: ScanGeometry, now_s: float) -> AxisPrecision:
        tangent = self.off_axis(other)
        degrees = float(np.degrees(np.arctan(tangent)))
        edge_px = abs(tangent) * pixels_along(scan, self.axis) / 2.0
        return AxisPrecision(now_s, self.axis, self.line.uncertainty, self.rounds, degrees, edge_px)


def measure_axis(session: Calibrating, axis: Axis) -> AxisFit:
    """One axis's line, measured to the precision its field's pixels ask for."""
    session.sink.log(AxisMeasuring(session.scope.now(), axis))
    line, taken, rounds = precise_line(session, axis, session.rough_scale(axis))
    measured = AxisFit(axis, line, taken, rounds)
    um_per_volt = measured.um_per_volt(session.scan)
    session.sink.log(AxisMeasured(session.scope.now(), axis, um_per_volt, line.scatter_um))
    return measured


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
    for axis, other in ((x, y), (y, x)):
        sink.log(axis.precision(other, scan, scope.now()))
    channel = scope.channel_labels[session.channel]
    return GalvoCalibration(
        x.um_per_volt(scan),
        y.um_per_volt(scan),
        x.line.scatter_um,
        y.line.scatter_um,
        channel,
        (*x.points(scan), *y.points(scan)),
    )
