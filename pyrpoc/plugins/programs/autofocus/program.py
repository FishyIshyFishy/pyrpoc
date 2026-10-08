"""Autofocus: scan one region at each z, score it by Tenengrad, and climb the
Prior's focus drive to the sharpest z. Each region frame is published as it is
taken, so the climb can be watched."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from pyrpoc.plugins.devices import DAQ, DaqError, Galvo, PriorError, PriorStage
from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.plugins.programs.program import (
    Cancelled,
    Program,
    RunContext,
    program_registry,
)

from ..building_blocks.parameter_groups import DaqGroup, FocusSearchGroup, RoiGroup, ScanGroup
from ..building_blocks.runners import Single
from .climb import ClimbResult, ClimbStep, climb
from .roi_scan import RoiRaster
from .tenengrad import tenengrad


def move_z(stage: PriorStage, z_um: float, sleep: Callable[[float], None]) -> None:
    stage.move_z_to(z_um)
    stage.wait_until_stopped("z", sleep)


def return_to_start(stage: PriorStage, start_z_um: float) -> None:
    """Stop and go back to where the run started. Waits with ``time.sleep``:
    the run is already stopped, so ``ctx.sleep`` would raise at once."""
    stage.stop_smoothly()
    stage.wait_until_stopped("z", time.sleep)
    move_z(stage, start_z_um, time.sleep)


class FocusRun:
    """One climb's measurements and what it records of them."""

    def __init__(self, ctx: RunContext, stage: PriorStage, raster: RoiRaster, start_z_um: float):
        self.ctx = ctx
        self.stage = stage
        self.raster = raster
        self.history: dict[str, Any] = {"start_z_um": start_z_um, "steps": []}

    def measure(self, z_um: float) -> float:
        self.ctx.check_cancel()
        move_z(self.stage, z_um, self.ctx.sleep)
        frame = self.raster.frame()
        self.ctx.publish("intensity", frame, channels=self.raster.channel_labels)
        return tenengrad(frame, self.raster.weights)

    def report(self, step: ClimbStep) -> None:
        # Recorded every step, so a run cut short still says where it went.
        self.history["steps"].append(asdict(step))
        self.ctx.describe("intensity", autofocus=self.history)
        self.ctx.status(
            f"z {step.z_um:.2f} µm · tenengrad {step.metric:.4g} · step {step.step_um:g} µm"
        )

    def finish(self, result: ClimbResult) -> None:
        self.history.update(
            best_z_um=result.best_z_um, best_metric=result.best_metric, at_edge=result.at_edge
        )
        self.ctx.describe("intensity", autofocus=self.history)
        move_z(self.stage, result.best_z_um, self.ctx.sleep)
        if result.at_edge:
            self.ctx.status(
                f"stopped at the search range's edge, z {result.best_z_um:.2f} µm: "
                "the focus may lie beyond it"
            )
        else:
            self.ctx.status(f"focused at z {result.best_z_um:.2f} µm")


@program_registry.register("autofocus")
class Autofocus(Program):
    display_name = "Autofocus"
    group = "Confocal Fluorescence"
    order = 13
    uses = [Galvo, DAQ, PriorStage]
    params = [ScanGroup, DaqGroup, RoiGroup, FocusSearchGroup]
    emits = {"intensity": Image2D}
    # Once only: each start is one complete search.
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        stage = ctx.devices[PriorStage]
        raster = RoiRaster.for_run(ctx)
        mask = ctx.params[RoiGroup].mask
        source = "center" if mask is None else mask.describe()
        ctx.describe("intensity", roi=raster.layout_metadata(source))
        start_z_um = stage.z_position()
        focus = FocusRun(ctx, stage, raster, start_z_um)
        try:
            result = climb(focus.measure, focus.report, start_z_um, ctx.params[FocusSearchGroup])
            focus.finish(result)
        except (Cancelled, DaqError, PriorError):
            return_to_start(stage, start_z_um)
            raise
