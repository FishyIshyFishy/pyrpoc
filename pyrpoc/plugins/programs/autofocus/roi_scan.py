"""A region scan: confocal's raster restricted to one connected region. Each
row is swept only from its first to its last lit pixel, with the scan's extra
steps either side for the mirrors to turn, so the scan is as short as the
region allows. Pitch and volts are the full frame's, so the region sits where
it was drawn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.structs.plugins.params import ParameterError
from pyrpoc.structs.plugins.programs.program import RunContext

from ..building_blocks.functions.daq_tasks import acquire
from ..building_blocks.functions.galvo_raster import pixel_samples
from ..building_blocks.parameter_groups import DaqGroup, RoiGroup, ScanGroup


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
    """The region's bounding box on the scan grid: the published frame."""

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


def centre_rectangle(scan: ScanGroup, fraction: float) -> np.ndarray:
    """A centred rectangle covering ``fraction`` of the grid's width and height."""
    height = max(1, round(scan.y_pixels * fraction))
    width = max(1, round(scan.x_pixels * fraction))
    top = (scan.y_pixels - height) // 2
    left = (scan.x_pixels - width) // 2
    region = np.zeros((scan.y_pixels, scan.x_pixels), dtype=bool)
    region[top : top + height, left : left + width] = True
    return region


def roi_region(roi: RoiGroup, scan: ScanGroup) -> np.ndarray:
    """The region as booleans on the scan grid. It must be one 8-connected
    area, so its rows are consecutive and the slow axis never jumps a gap."""
    if roi.mask is None:
        region = centre_rectangle(scan, roi.center_fraction)
    else:
        region = roi.mask.on_grid(scan.y_pixels, scan.x_pixels)
    # The count includes the background's label.
    labels, _ = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    if labels < 2:
        raise ParameterError("the region has no pixels on the scan grid")
    if labels > 2:
        raise ParameterError(f"the region must be one connected area, found {labels - 1}")
    return region


def row_spans(region: np.ndarray) -> list[RowSpan]:
    """Each lit row from its first lit pixel to its last. A gap inside a row is
    scanned rather than jumped, so every row is one sweep."""
    spans: list[RowSpan] = []
    for y in np.flatnonzero(region.any(axis=1)):
        lit = np.flatnonzero(region[y])
        spans.append(RowSpan(int(y), int(lit[0]), int(lit[-1])))
    return spans


def swept_columns(scan: ScanGroup, span: RowSpan) -> np.ndarray:
    """The columns one row sweeps: its span with the extra steps either side."""
    return np.arange(span.x0 - scan.extra_left, span.x1 + 1 + scan.extra_right)


def roi_waveform(scan: ScanGroup, spans: list[RowSpan], samples_per_pixel: int) -> np.ndarray:
    """The ``(fast, slow)`` AO waveform: each row's sweep in turn, holding
    ``samples_per_pixel`` per pixel."""
    columns = [swept_columns(scan, span) for span in spans]
    fast = np.concatenate([scan.fast_volts(sweep) for sweep in columns])
    rows = np.concatenate([np.full(len(columns[i]), span.y) for i, span in enumerate(spans)])
    slow = scan.slow_volts(rows)
    pixels = np.vstack((fast, slow)).astype(np.float64)
    return np.repeat(pixels, samples_per_pixel, axis=1)


def place_rows(
    raw: np.ndarray, scan: ScanGroup, spans: list[RowSpan], samples_per_pixel: int, box: Box
) -> np.ndarray:
    """A raw ``(channels, samples)`` stream as a ``(C, H, W)`` frame of ``box``:
    each pixel the mean of its samples, extra steps dropped, unscanned pixels 0."""
    channels = len(raw)
    frame = np.zeros((channels, box.height, box.width), dtype=np.float32)
    start = 0
    for span in spans:
        swept = span.width + scan.extra_left + scan.extra_right
        end = start + swept * samples_per_pixel
        pixels = raw[:, start:end].reshape(channels, swept, samples_per_pixel).mean(axis=2)
        left = span.x0 - box.left
        frame[:, span.y - box.top, left : left + span.width] = pixels[
            :, scan.extra_left : scan.extra_left + span.width
        ]
        start = end
    return frame


def metric_weights(region: np.ndarray, box: Box) -> np.ndarray:
    """The pixels whose whole 3x3 Sobel neighbourhood is inside the region, so
    the edge of the zero fill never counts as focus."""
    inside = box.crop(region).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(inside, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    weights = eroded > 0
    if not np.any(weights):
        raise ParameterError("the region is too thin to measure focus: it needs a 3x3 interior")
    return weights


@dataclass(frozen=True)
class RoiRaster:
    """One run's region scan. Built once: the region does not move between z."""

    daq: DAQ
    galvo: Galvo
    scan: ScanGroup
    sample_rate_hz: float
    samples_per_pixel: int
    spans: list[RowSpan]
    box: Box
    waveform: np.ndarray
    weights: np.ndarray

    @classmethod
    def for_run(cls, ctx: RunContext) -> RoiRaster:
        scan = ctx.params[ScanGroup]
        sample_rate_hz = ctx.params[DaqGroup].sample_rate_hz
        samples_per_pixel = pixel_samples(scan.dwell_time_us, sample_rate_hz)
        region = roi_region(ctx.params[RoiGroup], scan)
        spans = row_spans(region)
        box = Box.around(spans)
        return cls(
            ctx.devices[DAQ],
            ctx.devices[Galvo],
            scan,
            sample_rate_hz,
            samples_per_pixel,
            spans,
            box,
            roi_waveform(scan, spans, samples_per_pixel),
            metric_weights(region, box),
        )

    @property
    def channel_labels(self) -> list[str]:
        return [f"ai{index}" for index in self.daq.config.ai_channels]

    def frame(self) -> np.ndarray:
        """One region scan as a ``(C, H, W)`` frame of the bounding box."""
        raw = acquire(self.daq, self.galvo, self.waveform, {}, self.sample_rate_hz)
        return place_rows(raw, self.scan, self.spans, self.samples_per_pixel, self.box)

    def layout_metadata(self, source: str) -> dict[str, Any]:
        """Where the published frame sits on the full scan grid."""
        return {
            "source": source,
            "top": self.box.top,
            "left": self.box.left,
            "height": self.box.height,
            "width": self.box.width,
            "rows": len(self.spans),
            "pixels_swept": int(self.waveform.shape[1] // self.samples_per_pixel),
        }
