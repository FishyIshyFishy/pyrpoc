"""``RigMicroscope``: the galvo, DAQ and Prior stage, as a ``Microscope``."""

from __future__ import annotations

import time

import numpy as np

from pyrpoc.core.program import RunContext
from pyrpoc.plugins.devices import DAQ, Galvo, PriorStage
from pyrpoc.plugins.program_io.inputs import ScanGeometry, rate_hz

from ...functions.galvo_raster import Raster, pixel_samples, raster_waveform
from ..interfaces.microscope import DigitalLine


class RigMicroscope:
    """The rig a run claimed: rasters are built once per scan geometry and kept."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.daq = ctx.devices[DAQ]
        self.galvo = ctx.devices[Galvo]
        self.stage = ctx.devices[PriorStage]
        self.sample_rate_hz = rate_hz(ctx.inputs)
        self.rasters: dict[tuple[ScanGeometry, tuple[DigitalLine, ...]], Raster] = {}

    @property
    def channel_labels(self) -> list[str]:
        return [f"ai{index}" for index in self.daq.config.ai_channels]

    def raster(self, scan: ScanGeometry, lines: tuple[DigitalLine, ...]) -> Raster:
        key = (scan, lines)
        if key not in self.rasters:
            samples = pixel_samples(scan.dwell_time_us, self.sample_rate_hz)
            waveform = raster_waveform(scan, samples)
            device = self.daq.config.device_name
            high = np.ones(waveform.shape[1], dtype=bool)
            ttl = {f"{device}/port{each.port}/line{each.line}": high for each in lines}
            self.rasters[key] = Raster(
                self.daq, self.galvo, scan, self.sample_rate_hz, samples, waveform, ttl
            )
        return self.rasters[key]

    def scan(self, scan: ScanGeometry, lines: tuple[DigitalLine, ...]) -> np.ndarray:
        return self.raster(scan, lines).frame()

    def xy(self) -> tuple[float, float]:
        return self.stage.xy_position()

    def z(self) -> float:
        return self.stage.z_position()

    def move_xy(self, x_um: float, y_um: float) -> None:
        self.stage.move_xy_to(x_um, y_um)
        self.stage.wait_until_stopped("xy", self.ctx.sleep)

    def move_z(self, z_um: float) -> None:
        self.stage.move_z_to(z_um)
        self.stage.wait_until_stopped("z", self.ctx.sleep)

    def now(self) -> float:
        return time.monotonic()

    def wait(self, seconds: float) -> None:
        self.ctx.sleep(seconds)

    def stop(self) -> None:
        self.stage.stop_smoothly()
