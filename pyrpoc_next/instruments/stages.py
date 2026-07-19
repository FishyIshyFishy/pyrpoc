"""Motorized-stage placeholders (Prior, Zaber).

Not yet wired to hardware — they establish the interface a stage will expose:
a live position and a move command. A modality that positions the sample will use
these directly. ``position`` is the seam a gui widget can poll to live-update.
"""

from __future__ import annotations

from pyrpoc_next.instruments.base import Instrument
from pyrpoc_next.instruments.registry import instrument_registry
from pyrpoc_next.structs.keys import InstrumentKey


class Stage(Instrument):
    """Shared placeholder for a motorized XYZ stage."""

    def __init__(self):
        super().__init__()
        self.position = (0.0, 0.0, 0.0)

    def move_to(self, x: float, y: float, z: float) -> None:
        """Command an absolute move. Placeholder: records the target position."""
        self.position = (x, y, z)

    def summary(self) -> str:
        return f"{self.display_name}: {self.position}"


@instrument_registry.register
class PriorStage(Stage):
    """Prior motorized stage (placeholder)."""

    key = InstrumentKey.prior_stage
    display_name = "Prior Stage"


@instrument_registry.register
class ZaberStage(Stage):
    """Zaber motorized stage (placeholder)."""

    key = InstrumentKey.zaber_stage
    display_name = "Zaber Stage"
