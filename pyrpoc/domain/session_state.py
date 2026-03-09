from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .app_state import ParameterValue

SCHEMA_VERSION = 5


@dataclass
class InstrumentSessionState:
    type_key: str
    instance_id: str = ""
    connected: bool = False
    persisted_state: dict[str, Any] = field(default_factory=dict)
    config_values: list[ParameterValue] = field(default_factory=list)
    user_label: str | None = None


@dataclass
class OptoControlSessionState:
    type_key: str
    instance_id: str = ""
    connected: bool = False
    enabled: bool = False
    persisted_state: dict[str, Any] = field(default_factory=dict)
    config_values: list[ParameterValue] = field(default_factory=list)
    user_label: str | None = None


@dataclass
class DisplaySessionState:
    type_key: str
    instance_id: str = ""
    attached: bool = True
    dock_visible: bool = True
    persisted_state: dict[str, Any] = field(default_factory=dict)
    config_values: list[ParameterValue] = field(default_factory=list)
    user_label: str | None = None


@dataclass
class ModalitySessionState:
    selected_key: str | None = None
    configured_params: list[ParameterValue] = field(default_factory=list)


@dataclass
class SessionState:
    schema_version: int = SCHEMA_VERSION
    theme_mode: str = "system"
    instruments: list[InstrumentSessionState] = field(default_factory=list)
    optocontrols: list[OptoControlSessionState] = field(default_factory=list)
    displays: list[DisplaySessionState] = field(default_factory=list)
    modality: ModalitySessionState | None = None
