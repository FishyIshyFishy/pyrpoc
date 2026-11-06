"""Galvo calibration: the stage moves the sample by known distances, a clear feature is found
again in each new frame, and the fit of image shift against stage move gives µm per volt."""

# Each move goes out and comes back, and the two shifts are differenced, so anything that drifts
# steadily between frames (the cells, the stage, focus) cancels out of the measurement.

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

from ...analysis.features import Box, clear_feature
from ...analysis.noise import strongest_channel
from ...analysis.shifts import locate_near
from ...interfaces.microscope import Microscope
from .fit import CalibrationError, fit
from .sink import CalibrationSink

Axis = Literal["x", "y"]

FIRST_MOVE_UM = 1.0
GROWTH = 4.0
MAX_TRIES = 12
# A shift this large is measured to a few percent, which is enough to size the real moves.
ROUGH_SHIFT_PX = 4.0
# How far from its predicted place the feature is searched for: a fixed allowance for cells and
# stage drifting, plus a share of the predicted shift for the rough scale's error.
SEARCH_PX = 8.0
SEARCH_FRACTION = 0.25
# The feature must have this fraction of the view to move in, either way, along each axis.
ROOM_FRACTION = 0.125
# The real moves, as fractions of the room the feature has before it would leave the view.
MOVE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
EDGE_MARGIN_PX = 4


def volts_per_pixel(scan: ScanGeometry, axis: Axis) -> float:
    if axis == "x":
        return 2.0 * scan.fast_axis_amplitude / scan.x_pixels
    return 2.0 * scan.slow_axis_amplitude / scan.y_pixels


def room_px(feature: Box, shape: tuple[int, int], axis: Axis) -> float:
    """How far the feature can move either way along ``axis`` and stay in view."""
    height, width = shape
    if axis == "x":
        return min(feature.left, width - feature.right) - EDGE_MARGIN_PX
    return min(feature.top, height - feature.bottom) - EDGE_MARGIN_PX


@dataclass
class Calibrating:
    """One calibration in progress: the scan it takes, the feature it follows, and where the
    stage started."""

    scope: Microscope
    scan: ScanGeometry
    sink: CalibrationSink
    channel: int
    template: np.ndarray
    origin: tuple[float, float]
    # Where the template's corner was last found, in pixels.
    last: tuple[float, float]

    def take(self) -> np.ndarray:
        frame = self.scope.scan(self.scan, ())
        self.sink.show(frame)
        return frame

    def feature_at(self, expected: tuple[float, float], radius: float) -> tuple[float, float]:
        found = locate_near(self.template, self.take()[self.channel], expected, radius)
        if found is None:
            raise CalibrationError("lost the feature: it moved out of view or changed too much")
        self.last = found
        return found

    def shift_px(self, axis: Axis, move_um: float, predicted_px: float) -> float:
        """How far the feature moves along ``axis`` when the stage moves ``move_um``, where it
        is predicted to move ``predicted_px``."""
        index = 0 if axis == "x" else 1
        x, y = self.origin
        before = self.feature_at(self.last, SEARCH_PX)
        if axis == "x":
            self.scope.move_xy(x + move_um, y)
        else:
            self.scope.move_xy(x, y + move_um)
        expected = list(before)
        expected[index] += predicted_px
        radius = SEARCH_PX + SEARCH_FRACTION * abs(predicted_px)
        moved = self.feature_at((expected[0], expected[1]), radius)
        self.scope.move_xy(x, y)
        after = self.feature_at(before, radius)
        return ((moved[index] - before[index]) - (after[index] - moved[index])) / 2.0

    def log_move(self, axis: Axis, move_um: float, shift_px: float, rough: bool) -> None:
        self.sink.log(StageMoved(self.scope.now(), axis, move_um, shift_px, rough))

    def rough_scale(self, axis: Axis) -> float:
        """Pixels of shift per µm of move, from moves that grow until the shift is clear."""
        move, shift = FIRST_MOVE_UM, 0.0
        for _ in range(MAX_TRIES):
            # Unknown at first; after that, the last shift grown with the move.
            shift = self.shift_px(axis, move, shift * GROWTH)
            self.log_move(axis, move, shift, True)
            if abs(shift) >= ROUGH_SHIFT_PX:
                return shift / move
            move *= GROWTH
        raise CalibrationError(
            f"the stage moved {move / GROWTH:g} µm in {axis} but the image did not"
        )


def measure_axis(
    session: Calibrating, axis: Axis, room: float
) -> tuple[float, float, list[CalibrationPoint]]:
    """One axis's µm per volt, its residual, and the points it was fitted to."""
    session.sink.log(AxisMeasuring(session.scope.now(), axis))
    rough = session.rough_scale(axis)
    moves, shifts = [], []
    for fraction in MOVE_FRACTIONS:
        for sign in (1.0, -1.0):
            move = sign * fraction * room / abs(rough)
            shift = session.shift_px(axis, move, rough * move)
            session.log_move(axis, move, shift, False)
            moves.append(move)
            shifts.append(shift)
    slope, residual = fit(np.asarray(moves), np.asarray(shifts))
    per_px = volts_per_pixel(session.scan, axis)
    # The stage carrying the sample one way moves its image the other.
    um_per_volt = -1.0 / (slope * per_px)
    points = [CalibrationPoint(axis, m, s * per_px) for m, s in zip(moves, shifts, strict=True)]
    session.sink.log(AxisMeasured(session.scope.now(), axis, um_per_volt, residual))
    return um_per_volt, residual, points


def start_calibrating(
    scope: Microscope, scan: ScanGeometry, sink: CalibrationSink
) -> tuple[Calibrating, Box]:
    """Find the feature to follow in a first frame."""
    frame = scope.scan(scan, ())
    sink.show(frame)
    channel = strongest_channel(frame)
    plane = frame[channel]
    height, width = plane.shape
    margin_y, margin_x = int(height * ROOM_FRACTION), int(width * ROOM_FRACTION)
    roomy = Box(margin_y, margin_x, height - margin_y, width - margin_x)
    feature = clear_feature(plane, roomy)
    if feature is None:
        raise CalibrationError(
            "no clear feature in view away from its edges: move to where something bright "
            "is in view and try again"
        )
    region = feature.grown(EDGE_MARGIN_PX, (height, width))
    template = region.crop(plane)
    origin = scope.xy()
    corner = (float(region.left), float(region.top))
    sink.log(FeatureFound(scope.now(), scope.channel_labels[channel]))
    return Calibrating(scope, scan, sink, channel, template, origin, corner), feature


def calibrate(scope: Microscope, scan: ScanGeometry, sink: CalibrationSink) -> GalvoCalibration:
    """Measure both axes with ``scan``, showing every frame taken in ``sink``."""
    session, feature = start_calibrating(scope, scan, sink)
    shape = (scan.y_pixels, scan.x_pixels)
    try:
        x = measure_axis(session, "x", room_px(feature, shape, "x"))
        y = measure_axis(session, "y", room_px(feature, shape, "y"))
    except BaseException:
        scope.stop()
        raise
    channel = scope.channel_labels[session.channel]
    return GalvoCalibration(x[0], y[0], x[1], y[1], channel, (*x[2], *y[2]))
