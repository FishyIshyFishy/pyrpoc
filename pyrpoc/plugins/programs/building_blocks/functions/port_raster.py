"""The galvo raster with all of port 0 driven as one, as the LabVIEW imager drives it: the
masks' lines and the status lines packed into one word per sample, and static lines set
before the scan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.core.program_input import ParameterError
from pyrpoc.plugins.devices import DAQ, Galvo
from pyrpoc.plugins.program_io.inputs import (
    ACTIVE_LINE,
    IN_IMAGE_LINE,
    MASK_LINES,
    MASKS,
    NEW_LINE_LINE,
    NEW_PIXEL_LINE,
    STATUS_LINES,
    lines_for,
)
from pyrpoc.plugins.programs.building_blocks.settings import ScanGeometry, rate_hz

from .daq_tasks import acquire_port
from .galvo_raster import pixel_samples, raster_waveform
from .mask_ttl import mask_ttl

# The LabVIEW imager's "# New Line Ticks".
NEW_LINE_SAMPLES = 10
# Port 0 is the only hardware-timed port, and its words are 32 bits.
CLOCKED_PORT = 0
PORT_WIDTH = 32


def in_image_columns(scan: ScanGeometry) -> np.ndarray:
    """Which scanned columns are on the image rather than the extra steps."""
    columns = np.zeros(scan.total_x, dtype=bool)
    columns[scan.kept_columns] = True
    return columns


def status_signals(scan: ScanGeometry, samples_per_pixel: int) -> dict[str, np.ndarray]:
    """Active, In Image, New Line and New Pixel, each a flat boolean per raster sample."""
    columns = in_image_columns(scan)
    row_samples = scan.total_x * samples_per_pixel
    new_line = np.zeros(row_samples, dtype=bool)
    new_line[:NEW_LINE_SAMPLES] = True
    pixel_start = np.arange(samples_per_pixel) < -(-samples_per_pixel // 2)
    rows = {
        ACTIVE_LINE.key: np.ones(row_samples, dtype=bool),
        IN_IMAGE_LINE.key: np.repeat(columns, samples_per_pixel),
        NEW_LINE_LINE.key: new_line,
        NEW_PIXEL_LINE.key: np.outer(columns, pixel_start).reshape(-1),
    }
    return {key: np.tile(row, scan.y_pixels) for key, row in rows.items()}


def status_ttl(
    ctx: RunContext, scan: ScanGeometry, samples_per_pixel: int
) -> dict[tuple[int, int], np.ndarray]:
    """The status signals of the lines switched on, by (port, line)."""
    signals = status_signals(scan, samples_per_pixel)
    lines = {each.key: ctx.inputs[each] for each in STATUS_LINES}
    return {(line.port, line.line): signals[key] for key, line in lines.items() if line.on}


def mask_ttl_by_line(
    ctx: RunContext, scan: ScanGeometry, samples_per_pixel: int, device: str
) -> dict[tuple[int, int], np.ndarray]:
    """The masks' TTL, by (port, line)."""
    masks = ctx.inputs[MASKS]
    lines = lines_for(masks, ctx.inputs[MASK_LINES])
    by_channel = {line.channel(device): (line.port, line.line) for line in lines}
    ttl = mask_ttl(masks, lines, scan, samples_per_pixel, device)
    return {by_channel[channel]: signal for channel, signal in ttl.items()}


def check_lines(mask_lines: list[tuple[int, int]], status_lines: list[tuple[int, int]]) -> None:
    """Every line on port 0, within its width, and driven by one signal only."""
    every = mask_lines + status_lines
    for port, line in every:
        if port != CLOCKED_PORT or line >= PORT_WIDTH:
            raise ParameterError(
                f"p{port}.{line}: masks and status lines must be on port {CLOCKED_PORT}, "
                f"lines 0-{PORT_WIDTH - 1}"
            )
    shared = {each for each in every if every.count(each) > 1}
    if shared:
        names = ", ".join(f"p{port}.{line}" for port, line in sorted(shared))
        raise ParameterError(f"{names}: driven by more than one mask or status line")


def port_words(signals: dict[tuple[int, int], np.ndarray], total_samples: int) -> np.ndarray:
    """Every line's signal packed into one 32-bit word per sample, line ``n`` at bit ``n``."""
    words = np.zeros(total_samples, dtype=np.uint32)
    for (_, line), signal in signals.items():
        words |= signal.astype(np.uint32) << np.uint32(line)
    return words


@dataclass(frozen=True)
class PortRaster:
    """One run's scan: its waveform and the words played on port 0."""

    # Built once, before the first frame: they are the same every frame.

    daq: DAQ
    galvo: Galvo
    scan: ScanGeometry
    sample_rate_hz: float
    samples_per_pixel: int
    waveform: np.ndarray
    port_channel: str
    words: np.ndarray

    @classmethod
    def for_run(cls, ctx: RunContext) -> PortRaster:
        scan = ScanGeometry.from_inputs(ctx.inputs)
        sample_rate_hz = rate_hz(ctx.inputs)
        daq = ctx.devices[DAQ]
        device = daq.config.device_name
        samples_per_pixel = pixel_samples(scan.dwell_time_us, sample_rate_hz)
        masks = mask_ttl_by_line(ctx, scan, samples_per_pixel, device)
        statuses = status_ttl(ctx, scan, samples_per_pixel)
        check_lines(list(masks), list(statuses))
        waveform = raster_waveform(scan, samples_per_pixel)
        words = port_words({**masks, **statuses}, waveform.shape[1])
        port_channel = f"{device}/port{CLOCKED_PORT}"
        return cls(
            daq,
            ctx.devices[Galvo],
            scan,
            sample_rate_hz,
            samples_per_pixel,
            waveform,
            port_channel,
            words,
        )

    @property
    def channel_labels(self) -> list[str]:
        return [f"ai{index}" for index in self.daq.config.ai_channels]

    def frame(self) -> np.ndarray:
        """One scan as a ``(C, H, W)`` frame, each pixel the mean of its samples, with the
        overscan columns dropped."""
        raw = acquire_port(
            self.daq,
            self.galvo,
            self.waveform,
            self.port_channel,
            self.words,
            self.sample_rate_hz,
        )
        scan = self.scan
        cube = raw.astype(np.float32).reshape(
            len(raw), scan.y_pixels, scan.total_x, self.samples_per_pixel
        )
        return cube[:, :, scan.kept_columns, :].mean(axis=3)
