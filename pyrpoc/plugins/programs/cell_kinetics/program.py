"""Cell decay kinetics: how fast fluorescence fades where the RPOC laser
treated, measured against fluorescence imaged alongside it untreated.

The analysis of the TW-RPOC CellKineticsSuite, run on a series already in the
Data Library, so it uses no devices. Every channel is analysed on its own: its
first frame splits its pixels into fluorescent and background, and the mask
splits the fluorescent ones into treated and control. Each region's mean, less
the background's, is normalised to the first frame (F/F0), and the treated
trace is fitted with an exponential decay, corrected by the control's.

``decay`` holds each channel's fitted trace, its fit and its control, one
curve each, with the rate in the fit's label; the numbers are in its metadata.
``regions`` holds each channel's first frame beside its treated and control
pixels, for checking the split in the Overlay panel.
"""

from __future__ import annotations

import numpy as np

from pyrpoc.structs.data_library.data import Image2D, Spectrum1D
from pyrpoc.structs.plugins.programs.program import Program, RunContext, program_registry

from ..building_blocks.parameter_groups import DecayFitGroup, SeriesGroup, TreatmentGroup
from ..building_blocks.parameter_groups.decay_fit import DOUBLE_FIXED
from ..building_blocks.runners import Single
from .fitting import MIN_FRAMES, Fit, fit_decay
from .traces import Regions, normalized


def split_channels(
    ctx: RunContext, stack: np.ndarray, channels: tuple[str, ...], mask: np.ndarray
) -> list[Regions]:
    """Each channel's regions, published before any check so a split that
    stops the run can still be looked at."""
    every = [Regions.split(stack[0, index], mask) for index in range(len(channels))]
    layers: list[np.ndarray] = []
    names: list[str] = []
    for index, (name, regions) in enumerate(zip(channels, every, strict=True)):
        layers += [stack[0, index], regions.treated, regions.control]
        names += [name, f"{name} treated", f"{name} control"]
    ctx.publish("regions", np.stack(layers), channels=names)
    for name, regions in zip(channels, every, strict=True):
        if not regions.treated.any():
            raise ValueError(f"no fluorescent {name} pixels lie inside the mask")
    return every


def curves_for(
    name: str, data: np.ndarray, fit: Fit | None, control: np.ndarray | None, ratio: bool, px: int
) -> tuple[list[np.ndarray], list[str]]:
    """One channel's curves and their legend labels."""
    curves = [data]
    labels = [f"{name} treated" + (" ÷ control" if ratio else "")]
    if fit is not None:
        curves.append(fit.curve)
        labels.append(f"{name} fit, k={fit.rates['k']:.4f}/frame")
    if control is not None:
        curves.append(control)
        labels.append(f"{name} control ({px} px)")
    return curves, labels


def fit_channels(
    ctx: RunContext,
    stack: np.ndarray,
    channels: tuple[str, ...],
    every: list[Regions],
    model: str,
) -> None:
    """Every channel's decay, published as one curve per trace."""
    t = np.arange(len(stack), dtype=np.float64)
    curves: list[np.ndarray] = []
    labels: list[str] = []
    fits: dict[str, object] = {}
    for index, (name, regions) in enumerate(zip(channels, every, strict=True)):
        ctx.check_cancel()
        ctx.status(f"fitting {name}")
        frames = stack[:, index]
        treated = normalized(frames, regions.treated, regions.background)
        control = (
            normalized(frames, regions.control, regions.background)
            if regions.control.any()
            else None
        )
        data, fit = fit_decay(model, t, treated, control)
        ratio = control is not None and model != DOUBLE_FIXED
        more, named = curves_for(name, data, fit, control, ratio, int(regions.control.sum()))
        curves += more
        labels += named
        result = {"converged": False} if fit is None else fit.describe()
        fits[name] = {**result, **regions.describe()}
    ctx.describe("decay", model=model, fits=fits)
    ctx.publish("decay", np.stack(curves), channels=labels)


@program_registry.register("cell_kinetics")
class CellKinetics(Program):
    display_name = "Cell Decay Kinetics"
    group = "Analysis"
    order = 90
    uses = []
    params = [SeriesGroup, TreatmentGroup, DecayFitGroup]
    emits = {"decay": Spectrum1D, "regions": Image2D}
    # Once per start: the series is read as it stands when the run begins.
    runners = [Single()]

    def run(self, ctx: RunContext) -> None:
        series = ctx.params[SeriesGroup].series
        treated = ctx.params[TreatmentGroup].mask
        model = ctx.params[DecayFitGroup].model
        # Narrows for the type checker: both fields refuse to start a run unset.
        if series is None or treated is None:
            raise ValueError("a series and a treated mask are needed")
        stack = series.stack()
        if len(stack) < MIN_FRAMES[model]:
            raise ValueError(
                f"'{model}' needs at least {MIN_FRAMES[model]} frames; the series has {len(stack)}"
            )
        mask = treated.on_grid(*stack.shape[2:])
        every = split_channels(ctx, stack, series.channels, mask)
        fit_channels(ctx, stack, series.channels, every, model)
