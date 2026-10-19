"""Autofocus's region scan: the shared region geometry, read back as a frame of
its bounding box so it can be scored for focus."""

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
from ..building_blocks.functions.roi_scan import (
    Box,
    RowSpan,
    region_layout,
    roi_region,
    roi_waveform,
    row_spans,
)
from ..building_blocks.parameter_groups import DaqGroup, RoiGroup, ScanGroup


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
        pixels_swept = int(self.waveform.shape[1] // self.samples_per_pixel)
        return region_layout(source, self.spans, self.box, pixels_swept)
