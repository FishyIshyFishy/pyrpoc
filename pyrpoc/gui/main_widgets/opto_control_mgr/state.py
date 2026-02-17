from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtWidgets import QWidget

from pyrpoc.backend_utils.contracts import Action, Parameter
from pyrpoc.gui.main_widgets.opto_control_mgr.instance_card import InstanceCardWidget


@dataclass
class OptoControlManagerState:
    expanded_instance_id: str | None = None
    card_widgets: dict[str, InstanceCardWidget] = field(default_factory=dict)
    config_widgets_by_instance: dict[str, dict[str, QWidget]] = field(default_factory=dict)
    config_params_by_instance: dict[str, dict[str, Parameter]] = field(default_factory=dict)
    action_widgets_by_instance: dict[str, dict[str, dict[str, QWidget]]] = field(default_factory=dict)
    actions_by_label_by_instance: dict[str, dict[str, Action]] = field(default_factory=dict)
    method_to_action_by_instance: dict[str, dict[str, Action]] = field(default_factory=dict)
    enable_guard_by_instance: dict[str, bool] = field(default_factory=dict)
