from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyrpoc.gui.main_widgets.instance_card import RemovableCardWidget as InstanceCardWidget

if TYPE_CHECKING:
    from pyrpoc.backend_utils.parameter_utils import BaseParameter


@dataclass
class InstrumentManagerState:
    card_widgets: dict[object, InstanceCardWidget] = field(default_factory=dict)
    config_widgets: dict[str, tuple["BaseParameter", object]] = field(default_factory=dict)
    config_defs: dict[str, "BaseParameter"] = field(default_factory=dict)
    action_widgets: dict[str, dict[str, tuple["BaseParameter", object]]] = field(default_factory=dict)
    actions_by_label: dict[str, object] = field(default_factory=dict)
