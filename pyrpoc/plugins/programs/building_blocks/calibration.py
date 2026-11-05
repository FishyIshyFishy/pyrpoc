"""Galvo calibration: the stage moves the sample by known distances, a clear feature is found
again in each new frame, and the fit of image shift against stage move gives µm per volt."""

# Each move goes out and comes back, and the two shifts are differenced, so anything that drifts
# steadily between frames (the cells, the stage, focus) cancels out of the measurement.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.plugins.program_io.inputs import ScanGeometry
from pyrpoc.plugins.program_io.outputs import CalibrationPoint, GalvoCalibration, Image

from .microscope import Microscope
from .vision.features import Box, clear_feature
from .vision.noise import strongest_channel
from .vision.shifts import locate_near

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
# A point further than this many median deviations from the fit is a mismatch, not a measure.
OUTLIER_DEVIATIONS = 5.0
# A fit whose residual is more than this fraction of the largest move did not measure a line.
WORST_RESIDUAL = 0.02
# The real moves, as fractions of the room the feature has before it would leave the view.
MOVE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
EDGE_MARGIN_PX = 4


class CalibrationError(Exception):
    """The calibration could not be measured."""


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
    show: Callable[[np.ndarray], None]
    channel: int
    template: np.ndarray
    origin: tuple[float, float]
    # Where the template's corner was last found, in pixels.
    last: tuple[float, float]

    def take(self) -> np.ndarray:
        frame = self.scope.scan(self.scan, ())
        self.show(frame)
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

    def rough_scale(self, axis: Axis) -> float:
        """Pixels of shift per µm of move, from moves that grow until the shift is clear."""
        move, shift = FIRST_MOVE_UM, 0.0
        for _ in range(MAX_TRIES):
            # Unknown at first; after that, the last shift grown with the move.
            shift = self.shift_px(axis, move, shift * GROWTH)
            if abs(shift) >= ROUGH_SHIFT_PX:
                return shift / move
            move *= GROWTH
        raise CalibrationError(
            f"the stage moved {move / GROWTH:g} µm in {axis} but the image did not"
        )


def fit(moves: np.ndarray, shifts: np.ndarray) -> tuple[float, float]:
    """The slope through the origin of shift against move, refitted without mismatched
    points, and the residual in µm."""
    keep = np.ones(len(moves), dtype=bool)
    for _ in range(2):
        slope = float(np.sum(moves[keep] * shifts[keep]) / np.sum(moves[keep] ** 2))
        errors = np.abs(shifts - slope * moves)
        spread = float(np.median(errors[keep])) * 1.4826 + 1e-6
        keep = errors <= OUTLIER_DEVIATIONS * spread
    slope = float(np.sum(moves[keep] * shifts[keep]) / np.sum(moves[keep] ** 2))
    residual = float(np.sqrt(np.mean((shifts[keep] - slope * moves[keep]) ** 2))) / abs(slope)
    if residual > WORST_RESIDUAL * float(np.max(np.abs(moves))) or keep.sum() < len(moves) // 2:
        raise CalibrationError("the measured shifts do not fall on a line: try another spot")
    return slope, residual


def measure_axis(
    session: Calibrating, axis: Axis, room: float, status: Callable[[str], None]
) -> tuple[float, float, list[CalibrationPoint]]:
    """One axis's µm per volt, its residual, and the points it was fitted to."""
    rough = session.rough_scale(axis)
    moves, shifts = [], []
    for fraction in MOVE_FRACTIONS:
        for sign in (1.0, -1.0):
            move = sign * fraction * room / abs(rough)
            shift = session.shift_px(axis, move, rough * move)
            status(f"{axis}: moved {move:+.2f} µm, image shifted {shift:+.2f} px")
            moves.append(move)
            shifts.append(shift)
    slope, residual = fit(np.asarray(moves), np.asarray(shifts))
    per_px = volts_per_pixel(session.scan, axis)
    # The stage carrying the sample one way moves its image the other.
    um_per_volt = -1.0 / (slope * per_px)
    points = [CalibrationPoint(axis, m, s * per_px) for m, s in zip(moves, shifts, strict=True)]
    return um_per_volt, residual, points


def start_calibrating(
    scope: Microscope, scan: ScanGeometry, show: Callable[[np.ndarray], None]
) -> tuple[Calibrating, Box]:
    """Find the feature to follow in a first frame."""
    frame = scope.scan(scan, ())
    show(frame)
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
    return Calibrating(scope, scan, show, channel, template, origin, corner), feature


def calibrate(ctx: RunContext, scope: Microscope, scan: ScanGeometry) -> None:
    """Measure both axes, publishing every frame as ``view`` and the result as
    ``calibration``."""

    def show(frame: np.ndarray) -> None:
        ctx.publish("view", Image(frame, scope.channel_labels))

    session, feature = start_calibrating(scope, scan, show)
    shape = (scan.y_pixels, scan.x_pixels)
    try:
        x = measure_axis(session, "x", room_px(feature, shape, "x"), ctx.status)
        y = measure_axis(session, "y", room_px(feature, shape, "y"), ctx.status)
    except BaseException:
        scope.stop()
        raise
    result = GalvoCalibration(
        x[0], y[0], x[1], y[1], scope.channel_labels[session.channel], (*x[2], *y[2])
    )
    ctx.publish("calibration", result)
    ctx.status(f"{result.um_per_volt_x:.3f} × {result.um_per_volt_y:.3f} µm per volt")
