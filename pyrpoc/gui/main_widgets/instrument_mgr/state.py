from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtWidgets import QWidget

from pyrpoc.backend_utils.contracts import Action


@dataclass
class InstrumentManagerState:
    config_widgets: dict[str, QWidget] = field(default_factory=dict)
    action_widgets: dict[str, dict[str, QWidget]] = field(default_factory=dict)
    actions_by_label: dict[str, Action] = field(default_factory=dict)
