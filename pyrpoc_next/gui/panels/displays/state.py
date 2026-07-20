"""Per-widget UI state for the display manager: the card map."""

from __future__ import annotations

from dataclasses import dataclass, field

from pyrpoc_next.gui.widgets.cards import RemovableCardWidget


@dataclass
class DisplayManagerState:
    """Maps each live display to its card, so refreshes reuse cards."""

    card_widgets: dict[object, RemovableCardWidget] = field(default_factory=dict)
