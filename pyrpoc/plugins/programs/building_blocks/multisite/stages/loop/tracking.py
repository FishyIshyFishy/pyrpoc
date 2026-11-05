"""Tracking: finding each unit again near where it was last seen, in a frame centred there, by
matching the image of it last time. Tracking moves a unit's outline; nothing else does."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pyrpoc.plugins.program_io.outputs import MotionRate, Unit

from ...analysis.noise import strongest_channel
from ...analysis.shifts import locate_near
from ...interfaces.field import FieldOfView
from ...interfaces.overview import Overview

# Room kept around a unit in its template and its frames, for its edges and blur.
MARGIN_UM = 5.0
# How many spreads of the motion rate the search allows beyond the median.
SPREAD_ALLOWANCE = 2.0


@dataclass
class Track:
    """Where one unit is now: its outline as tracked, the image last matched (None until the
    next scan, when the unit is taken to be where its outline says), on which channel, how far
    that image's centre sits from the unit's in pixels (x, y), and when it was last seen."""

    unit: Unit
    template: np.ndarray | None
    channel: int
    template_offset_px: tuple[float, float]
    seen_s: float

    @property
    def centre(self) -> tuple[float, float]:
        return self.unit.centre


@dataclass(frozen=True)
class Found:
    """Where a unit was found: how far it moved in µm, and how well it matched."""

    shift_um: tuple[float, float]
    radius_um: float


def window_px(unit: Unit, um_per_px: float) -> tuple[int, int]:
    """The (height, width) in pixels of a unit's frames: its bounds and a margin."""
    left, top, right, bottom = unit.bounds()
    width = int(np.ceil((right - left + 2 * MARGIN_UM) / um_per_px))
    height = int(np.ceil((bottom - top + 2 * MARGIN_UM) / um_per_px))
    return height, width


def window_corner(centre_px: tuple[float, float], size: tuple[int, int]) -> tuple[int, int]:
    """The whole pixel (left, top) at which a ``size`` (height, width) window most nearly
    centred on (x, y) ``centre_px`` starts."""
    height, width = size
    return int(round(centre_px[0] - width / 2.0)), int(round(centre_px[1] - height / 2.0))


def crop_centred(
    frame: np.ndarray, centre_px: tuple[float, float], size: tuple[int, int]
) -> np.ndarray:
    """The ``size`` (height, width) window of a ``(C, H, W)`` frame centred, to the nearest
    pixel, on (x, y) ``centre_px``, zero where it runs past the frame."""
    height, width = size
    left, top = window_corner(centre_px, size)
    out = np.zeros((frame.shape[0], height, width), dtype=np.float32)
    src_top, src_left = max(0, top), max(0, left)
    src_bottom = min(frame.shape[1], top + height)
    src_right = min(frame.shape[2], left + width)
    if src_bottom > src_top and src_right > src_left:
        out[:, src_top - top : src_bottom - top, src_left - left : src_right - left] = frame[
            :, src_top:src_bottom, src_left:src_right
        ]
    return out


def outline_mask(unit: Unit, size: tuple[int, int], um_per_px: float) -> np.ndarray:
    """The unit's outline filled, in a window of ``size`` centred on it."""
    height, width = size
    cx, cy = unit.centre
    points = np.array(
        [
            ((x - cx) / um_per_px + width / 2.0, (y - cy) / um_per_px + height / 2.0)
            for x, y in unit.outline
        ]
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask > 0


def overview_template(unit: Unit, overview: Overview, um_per_px: float) -> tuple[np.ndarray, int]:
    """The unit's window cut from the survey's overview at the loop's pixel size, and the
    channel it shows best in."""
    height, width = window_px(unit, um_per_px)
    cx, cy = unit.centre
    placement = overview.placement
    scale = placement.um_per_px
    centre_px = ((cx - placement.left_um) / scale, (cy - placement.top_um) / scale)
    size = (int(round(height * um_per_px / scale)), int(round(width * um_per_px / scale)))
    cut = crop_centred(overview.planes, centre_px, size)
    resized = np.stack(
        [cv2.resize(plane, (width, height), interpolation=cv2.INTER_AREA) for plane in cut]
    )
    channel = strongest_channel(resized)
    return resized[channel], channel


class Tracker:
    """Finds units in frames centred where each was last seen."""

    def __init__(self, field: FieldOfView, motion: MotionRate | None):
        self.field = field
        self.motion = motion

    def radius_um(self, elapsed_s: float) -> float:
        """How far a unit may have moved in ``elapsed_s``."""
        if self.motion is None:
            return 2.0 * MARGIN_UM
        speed = self.motion.um_per_min + SPREAD_ALLOWANCE * self.motion.spread_um_per_min
        return speed * elapsed_s / 60.0 + MARGIN_UM

    def locate(self, track: Track, frame: np.ndarray, now_s: float) -> Found | None:
        """Where the unit is in ``frame``, a field centred on its last centre, or None when it
        is not found within reach."""
        radius_um = self.radius_um(now_s - track.seen_s)
        if track.template is None:
            return Found((0.0, 0.0), radius_um)
        pixels = self.field.pixels
        t_height, t_width = track.template.shape
        offset_x, offset_y = track.template_offset_px
        expected = ((pixels - t_width) / 2.0 + offset_x, (pixels - t_height) / 2.0 + offset_y)
        room = max(0.0, (pixels - max(t_height, t_width)) / 2.0)
        radius_px = min(radius_um / self.field.um_per_px, room)
        found = locate_near(track.template, frame[track.channel], expected, radius_px)
        if found is None:
            return None
        shift = (
            (found[0] - expected[0]) * self.field.um_per_px,
            (found[1] - expected[1]) * self.field.um_per_px,
        )
        return Found(shift, radius_um)

    def follow(self, track: Track, found: Found, frame: np.ndarray, now_s: float) -> None:
        """Move the unit by what was found, and take its new look as the template."""
        track.unit = track.unit.moved_by(*found.shift_um)
        centre = self.field.pixels / 2.0
        shifted = (
            centre + found.shift_um[0] / self.field.um_per_px,
            centre + found.shift_um[1] / self.field.um_per_px,
        )
        if track.template is None:
            height, width = window_px(track.unit, self.field.um_per_px)
            track.channel = strongest_channel(crop_centred(frame, shifted, (height, width)))
        else:
            height, width = track.template.shape
        plane = frame[track.channel : track.channel + 1]
        track.template = crop_centred(plane, shifted, (height, width))[0]
        # The crop starts on a whole pixel, so its centre misses the unit's by up to half a
        # pixel; that is kept, so the rounding never builds up from one scan to the next.
        left, top = window_corner(shifted, (height, width))
        track.template_offset_px = (
            left + width / 2.0 - shifted[0],
            top + height / 2.0 - shifted[1],
        )
        track.seen_s = now_s


def start_track(unit: Unit, overview: Overview, field: FieldOfView, seen_s: float) -> Track:
    """A track starting from the survey, where the unit was drawn."""
    template, channel = overview_template(unit, overview, field.um_per_px)
    return Track(unit, template, channel, (0.0, 0.0), seen_s)


def placed_track(unit: Unit, seen_s: float) -> Track:
    """A track taking the unit to be where ``unit`` says at the next scan, and learning its look
    from that frame: for a unit redrawn where it is now, or carried on after a resume."""
    return Track(unit, None, 0, (0.0, 0.0), seen_s)
