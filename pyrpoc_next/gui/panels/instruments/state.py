"""Per-widget UI state for the instrument manager: the card map."""

from __future__ import annotations

from dataclasses import dataclass, field

from pyrpoc_next.gui.widgets.cards import RemovableCardWidget


@dataclass
class InstrumentManagerState:
    """Maps each live instrument instance to its card, so refreshes reuse cards."""

    card_widgets: dict[object, RemovableCardWidget] = field(default_factory=dict)
