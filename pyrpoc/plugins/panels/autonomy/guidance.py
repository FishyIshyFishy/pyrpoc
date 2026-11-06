"""What the Sites panel says on its status line: where the user is, and what to do next."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pyrpoc.plugins.program_io.outputs import LoopView, UnitState


class LoopStage(Enum):
    """Where the loop on the shown survey is."""

    NONE = "none"
    RUNNING = "running"
    STOPPED_EARLY = "stopped early"
    FINISHED = "finished"


def loop_stage(view: LoopView | None, running: bool) -> LoopStage:
    """A loop that ended with routines still open was stopped, and can be carried on."""
    if view is None:
        return LoopStage.NONE
    if running:
        return LoopStage.RUNNING
    open_work = any(unit.state not in (UnitState.DONE, UnitState.REMOVED) for unit in view.units)
    return LoopStage.STOPPED_EARLY if open_work else LoopStage.FINISHED


@dataclass(frozen=True)
class PanelState:
    """What the status line depends on."""

    has_survey: bool
    has_map: bool
    drawing: bool
    sites: int
    stage: LoopStage
    minutes: float


def duration(minutes: float) -> str:
    """A loop's running time in words."""
    if minutes < 1:
        return "under a minute"
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"


def guidance(state: PanelState) -> str:
    if not state.has_survey:
        return "Run a Survey first: its map of the dish is where you draw sites."
    if not state.has_map:
        return "The survey's map appears here when the survey finishes."
    if state.drawing:
        return (
            "Click each corner of the site on the map. Click the first corner again, or "
            "double-click, to close it. Press Esc to stop drawing."
        )
    return after_drawing(state)


def after_drawing(state: PanelState) -> str:
    sites = f"{state.sites} site" + ("" if state.sites == 1 else "s")
    minutes = duration(state.minutes)
    match state.stage:
        case LoopStage.NONE if state.sites == 0:
            return "Click Draw site, then click the corners of a region on the map to follow."
        case LoopStage.NONE:
            return f"{sites} drawn. Next: start Longitudinal Monitor in Acquisition to follow them."
        case LoopStage.RUNNING:
            return f"{sites} in the loop. Draw a site to add it; click one to pause it."
        case LoopStage.STOPPED_EARLY:
            return (
                f"The loop stopped early, {minutes} in. Start Longitudinal Monitor again "
                "with this folder to carry it on."
            )
        case LoopStage.FINISHED:
            return f"The loop finished, {minutes} in all. These are the {sites} it followed."
