"""A region scan's geometry: confocal's raster restricted to one connected region."""

# Each row is swept only from its first to its last lit pixel, with the scan's extra steps either
# side for the mirrors to turn, so the scan is as short as the region allows. Pitch and volts are
# the full frame's, so the region sits where it was drawn.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from pyrpoc.core.mask import Mask
from pyrpoc.core.program_inputs import ParameterError

from ..program_inputs import ScanGeometry


@dataclass(frozen=True)
class RowSpan:
    """One scanned row: ``y`` and its first and last lit columns, inclusive."""

    y: int
    x0: int
    x1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1


@dataclass(frozen=True)
class Box:
    """The region's bounding box on the scan grid."""

    top: int
    left: int
    height: int
    width: int

    @classmethod
    def around(cls, spans: list[RowSpan]) -> Box:
        left = min(span.x0 for span in spans)
        right = max(span.x1 for span in spans)
        return cls(spans[0].y, left, spans[-1].y - spans[0].y + 1, right - left + 1)

    def crop(self, plane: np.ndarray) -> np.ndarray:
        return plane[self.top : self.top + self.height, self.left : self.left + self.width]


def centre_rectangle(scan: ScanGeometry, fraction: float) -> np.ndarray:
    """A centred rectangle covering ``fraction`` of the grid's width and height."""
    height = max(1, round(scan.y_pixels * fraction))
    width = max(1, round(scan.x_pixels * fraction))
    top = (scan.y_pixels - height) // 2
    left = (scan.x_pixels - width) // 2
    region = np.zeros((scan.y_pixels, scan.x_pixels), dtype=bool)
    region[top : top + height, left : left + width] = True
    return region


def roi_region(masks: tuple[Mask, ...], center_fraction: float, scan: ScanGeometry) -> np.ndarray:
    """The drawn masks together as booleans on the scan grid, or with none, the centred
    rectangle ``center_fraction`` of the scan wide and high. It must be one 8-connected area."""
    # One area, so its rows are consecutive and the slow axis never jumps a gap.
    if not masks:
        region = centre_rectangle(scan, center_fraction)
    else:
        # Every drawn mask together, so a region drawn as several shapes is still one region.
        region = np.logical_or.reduce([m.on_grid(scan.y_pixels, scan.x_pixels) for m in masks])
    # The count includes the background's label.
    labels, _ = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    if labels < 2:
        raise ParameterError("the region has no pixels on the scan grid")
    if labels > 2:
        raise ParameterError(
            f"the region must be one connected area, found {labels - 1}; the drawn masks "
            "are scanned together, and a mask drawn with Apply thresholds on keeps only "
            "in-range pixels, which can split it"
        )
    return region


def region_source(masks: tuple[Mask, ...]) -> str:
    """What the region was drawn from, for the recording."""
    return ", ".join(mask.describe() for mask in masks) or "center"


def row_spans(region: np.ndarray) -> list[RowSpan]:
    """Each lit row, from its first lit pixel to its last."""
    # A gap inside a row is scanned rather than jumped, so every row is one sweep.
    spans: list[RowSpan] = []
    for y in np.flatnonzero(region.any(axis=1)):
        lit = np.flatnonzero(region[y])
        spans.append(RowSpan(int(y), int(lit[0]), int(lit[-1])))
    return spans


def swept_columns(scan: ScanGeometry, span: RowSpan) -> np.ndarray:
    """The columns one row sweeps: its span with the extra steps either side."""
    return np.arange(span.x0 - scan.extra_left, span.x1 + 1 + scan.extra_right)


def roi_waveform(scan: ScanGeometry, spans: list[RowSpan], samples_per_pixel: int) -> np.ndarray:
    """The ``(fast, slow)`` AO waveform: each row's sweep in turn, holding
    ``samples_per_pixel`` per pixel."""
    columns = [swept_columns(scan, span) for span in spans]
    fast = np.concatenate([scan.fast_volts(sweep) for sweep in columns])
    rows = np.concatenate([np.full(len(columns[i]), span.y) for i, span in enumerate(spans)])
    slow = scan.slow_volts(rows)
    pixels = np.vstack((fast, slow)).astype(np.float64)
    return np.repeat(pixels, samples_per_pixel, axis=1)


def region_layout(source: str, spans: list[RowSpan], box: Box, pixels_swept: int) -> dict[str, Any]:
    """Where the region sits on the full scan grid, for the recording."""
    return {
        "source": source,
        "top": box.top,
        "left": box.left,
        "height": box.height,
        "width": box.width,
        "rows": len(spans),
        "pixels_swept": pixels_swept,
    }
