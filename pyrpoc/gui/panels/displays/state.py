from __future__ import annotations

from dataclasses import dataclass, field

from pyrpoc.gui.panels.cards import RemovableCardWidget as DisplayCardWidget


@dataclass
class DisplayManagerState:
    card_widgets: dict[object, DisplayCardWidget] = field(default_factory=dict)
