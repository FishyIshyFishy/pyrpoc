from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyrpoc.displays.base_display import BaseDisplay
from pyrpoc.instruments.base_instrument import BaseInstrument
from pyrpoc.modalities.base_modality import BaseModality
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl


@dataclass
class ParameterValue:
    label: str
    value: Any


@dataclass(eq=False)
class InstrumentState:
    type_key: str
    instance: BaseInstrument
    connected: bool = False
    config_values: list[ParameterValue] = field(default_factory=list)
    last_error: str | None = None
    user_label: str | None = None


@dataclass(eq=False)
class OptoControlState:
    type_key: str
    instance: BaseOptoControl
    connected: bool = False
    enabled: bool = False
    config_values: list[ParameterValue] = field(default_factory=list)
    last_error: str | None = None
    user_label: str | None = None


@dataclass(eq=False)
class DisplayState:
    type_key: str
    instance: BaseDisplay
    attached: bool = True
    config_values: list[ParameterValue] = field(default_factory=list)
    last_error: str | None = None
    user_label: str | None = None


@dataclass
class ModalityState:
    selected_key: str | None = None
    selected_class: type[BaseModality] | None = None
    instance: BaseModality | None = None
    configured_params: list[ParameterValue] = field(default_factory=list)
    running: bool = False
    last_error: str | None = None


@dataclass
class GuiLayoutState:
    ads_state_base64: str | None = None
    dock_visibility: dict[str, bool] = field(default_factory=dict)
    expanded_opto_index: int | None = None


@dataclass
class AppState:
    instruments: list[InstrumentState] = field(default_factory=list)
    optocontrols: list[OptoControlState] = field(default_factory=list)
    displays: list[DisplayState] = field(default_factory=list)
    modality: ModalityState = field(default_factory=ModalityState)
    gui_layout: GuiLayoutState = field(default_factory=GuiLayoutState)
