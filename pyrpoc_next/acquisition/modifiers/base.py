"""Modifiers are inert configuration dataclasses. A modality realizes them.

The registry maps a ModifierKey to its dataclass so the core can build a modifier
from a routine's saved values.
"""

from __future__ import annotations

from typing import Any

from attrs import define

from pyrpoc_next.structs.keys import ModifierKey
from pyrpoc_next.structs.manifest import ModifierManifest


@define
class Modifier:
    """Base for a modifier's configuration. No behavior — realization is the modality's."""


class ModifierRegistry:
    """Maps a ModifierKey to its modifier dataclass."""

    def __init__(self):
        self.entries: dict[ModifierKey, type[Modifier]] = {}

    def register(self, cls: type[Modifier]) -> type[Modifier]:
        self.entries[cls.key] = cls
        return cls

    def manifest(self, key: ModifierKey) -> ModifierManifest:
        return self.entries[key].manifest

    def build(self, key: ModifierKey, values: dict[str, Any]) -> Modifier:
        return self.entries[key].from_values(values)

    def available(self) -> list[ModifierKey]:
        return list(self.entries)


modifier_registry = ModifierRegistry()


# A concrete modifier dataclass declares: key (ClassVar[ModifierKey]),
# manifest (ClassVar[ModifierManifest]), and a from_values(values) classmethod.
