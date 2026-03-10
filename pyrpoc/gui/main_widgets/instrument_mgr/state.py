from __future__ import annotations

from dataclasses import dataclass, field

from pyrpoc.gui.main_widgets.opto_control_mgr.instance_card import InstanceCardWidget


@dataclass
class InstrumentManagerState:
    card_widgets: dict[object, InstanceCardWidget] = field(default_factory=dict)
    config_widgets: dict[str, tuple["BaseParameter", object]] = field(default_factory=dict)
    config_defs: dict[str, "BaseParameter"] = field(default_factory=dict)
    action_widgets: dict[str, dict[str, tuple["BaseParameter", object]]] = field(default_factory=dict)
    actions_by_label: dict[str, object] = field(default_factory=dict)
