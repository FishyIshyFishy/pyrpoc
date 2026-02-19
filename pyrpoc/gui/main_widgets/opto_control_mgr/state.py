from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtWidgets import QLineEdit, QToolButton, QWidget

from pyrpoc.backend_utils.contracts import Action, Parameter
from pyrpoc.gui.main_widgets.opto_control_mgr.instance_card import InstanceCardWidget


@dataclass
class OptoControlManagerState:
    expanded_instance: object | None = None
    card_widgets: dict[object, InstanceCardWidget] = field(default_factory=dict)
    config_widgets_by_instance: dict[object, dict[str, QWidget]] = field(default_factory=dict)
    config_params_by_instance: dict[object, dict[str, Parameter]] = field(default_factory=dict)
    action_widgets_by_instance: dict[object, dict[str, dict[str, QWidget]]] = field(default_factory=dict)
    actions_by_label_by_instance: dict[object, dict[str, Action]] = field(default_factory=dict)
    method_to_action_by_instance: dict[object, dict[str, Action]] = field(default_factory=dict)
    enable_guard_by_instance: dict[object, bool] = field(default_factory=dict)
    active_mask_editor_instance: object | None = None
    active_mask_editor_widget: QWidget | None = None
    mask_path_widgets_by_instance: dict[object, QLineEdit] = field(default_factory=dict)
    mask_create_buttons_by_instance: dict[object, QToolButton] = field(default_factory=dict)
    mask_tempfile_by_instance: dict[object, Path] = field(default_factory=dict)
    ui_locked_for_editor: bool = False
