from __future__ import annotations

from dataclasses import dataclass, field

from .app_state import ParameterValue

SCHEMA_VERSION = 4


@dataclass
class InstrumentSessionState:
    type_key: str
    connected: bool = False
    config_values: list[ParameterValue] = field(default_factory=list)
    user_label: str | None = None


@dataclass
class OptoControlSessionState:
    type_key: str
    connected: bool = False
    enabled: bool = False
    config_values: list[ParameterValue] = field(default_factory=list)
    user_label: str | None = None


@dataclass
class DisplaySessionState:
    type_key: str
    attached: bool = True
    dock_visible: bool = True
    config_values: list[ParameterValue] = field(default_factory=list)
    user_label: str | None = None


@dataclass
class ModalitySessionState:
    selected_key: str | None = None
    configured_params: list[ParameterValue] = field(default_factory=list)


@dataclass
class GuiLayoutSessionState:
    ads_state_base64: str | None = None
    dock_visibility: dict[str, bool] = field(default_factory=dict)
    expanded_opto_index: int | None = None


@dataclass
class SessionState:
    schema_version: int = SCHEMA_VERSION
    theme_mode: str = "system"
    instruments: list[InstrumentSessionState] = field(default_factory=list)
    optocontrols: list[OptoControlSessionState] = field(default_factory=list)
    displays: list[DisplaySessionState] = field(default_factory=list)
    modality: ModalitySessionState | None = None
    gui_layout: GuiLayoutSessionState = field(default_factory=GuiLayoutSessionState)
