from __future__ import annotations

from dataclasses import dataclass, field

from pyrpoc.gui.main_widgets.opto_control_mgr.instance_card import InstanceCardWidget


@dataclass
class OptoControlManagerState:
    card_widgets: dict[object, InstanceCardWidget] = field(default_factory=dict)
