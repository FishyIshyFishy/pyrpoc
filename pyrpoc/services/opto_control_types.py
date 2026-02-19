from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyrpoc.backend_utils.contracts import Action, Parameter
from pyrpoc.domain.app_state import OptoControlState


@dataclass(frozen=True)
class OptoStatusSnapshot:
    last_action: str
    enabled: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class OptoInstanceSchema:
    type_key: str
    display_name: str
    config_parameters: dict[str, list[Parameter]]
    actions: tuple[Action, ...]
    editor_key: str | None
    editor_anchor_param: str | None
    editor_apply_method: str | None


@dataclass(frozen=True)
class OptoInstanceRow:
    state: OptoControlState
    key: str
    display_name: str
    connected: bool
    compatible_with_selected_modality: bool
    status: OptoStatusSnapshot


@dataclass(frozen=True)
class ActionExecutionResult:
    state: OptoControlState
    action_label: str
    status: OptoStatusSnapshot
