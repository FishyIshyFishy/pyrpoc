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


@dataclass
class ModalityState:
    selected_key: str | None = None
    selected_class: type[BaseModality] | None = None
    instance: BaseModality | None = None
    # Remembered raw parameter values keyed by modality key, so switching
    # modalities (and relaunching the app) preserves each modality's settings.
    params_by_modality: dict[str, list[ParameterValue]] = field(default_factory=dict)
    running: bool = False
    last_error: str | None = None

    @property
    def configured_params(self) -> list[ParameterValue]:
        """Remembered parameter values for the currently selected modality."""
        if self.selected_key is None:
            return []
        return self.params_by_modality.get(self.selected_key, [])


@dataclass
class GuiLayoutState:
    ads_state_base64: str | None = None
    dock_visibility: dict[str, bool] = field(default_factory=dict)
    expanded_opto_index: int | None = None


@dataclass
class AppState:
    # Instance-first inventory model:
    # - each list entry is the concrete runtime object
    # - service rows still expose this object via row["state"] for UI compatibility
    instruments: list[BaseInstrument] = field(default_factory=list)
    optocontrols: list[BaseOptoControl] = field(default_factory=list)
    displays: list[BaseDisplay] = field(default_factory=list)
    modality: ModalityState = field(default_factory=ModalityState)
    gui_layout: GuiLayoutState = field(default_factory=GuiLayoutState)
