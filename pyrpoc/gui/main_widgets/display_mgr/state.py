from __future__ import annotations

from dataclasses import dataclass, field

from pyrpoc.gui.main_widgets.instance_card import RemovableCardWidget as DisplayCardWidget


@dataclass
class DisplayManagerState:
    card_widgets: dict[object, DisplayCardWidget] = field(default_factory=dict)
