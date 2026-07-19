"""Routine: the user's inert, ordered recipe for a session.

A routine names modality blocks and, per block, which modifiers are available and
enabled. It holds no logic — the core interprets it. One block is "active" and is
what runs on play. Schema supports many blocks; the first UI cut uses one.
"""

from __future__ import annotations

from attrs import define, field

from pyrpoc_next.structs.keys import ModalityKey, ModifierKey
from pyrpoc_next.structs.parameters import ParameterValue


@define
class ModifierSlot:
    """A modifier offered in a block: its config, and whether it is available/enabled."""

    key: ModifierKey
    values: list[ParameterValue] = field(factory=list)
    available: bool = True
    enabled: bool = False


@define
class RoutineBlock:
    """One modality in a routine, with its parameter values and modifier slots."""

    modality: ModalityKey
    values: list[ParameterValue] = field(factory=list)
    modifiers: list[ModifierSlot] = field(factory=list)

    def enabled_modifiers(self) -> list[ModifierSlot]:
        return [slot for slot in self.modifiers if slot.available and slot.enabled]


@define
class Routine:
    """An ordered set of blocks with one marked active."""

    name: str = "untitled"
    blocks: list[RoutineBlock] = field(factory=list)
    active_index: int = 0

    @property
    def active_block(self) -> RoutineBlock | None:
        if 0 <= self.active_index < len(self.blocks):
            return self.blocks[self.active_index]
        return None
