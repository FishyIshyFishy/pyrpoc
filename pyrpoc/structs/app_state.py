from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyrpoc.structs.parameters import ParameterValue

if TYPE_CHECKING:
    from pyrpoc.instruments.base import BaseInstrument
    from pyrpoc.optocontrols.base import BaseOptoControl
    from pyrpoc.presets.base import Preset
    from pyrpoc.gui.displays.base_display import BaseDisplay


@dataclass
class PresetState:
    """Runtime selection state for the active preset.

    (Kept attribute name ``params_by_preset``; the GUI reaches this through
    ``AppState.preset``.)
    """

    selected_key: str | None = None
    selected_class: type[Preset] | None = None
    instance: Preset | None = None
    params_by_preset: dict[str, list[ParameterValue]] = field(default_factory=dict)
    running: bool = False
    last_error: str | None = None

    @property
    def configured_params(self) -> list[ParameterValue]:
        if self.selected_key is None:
            return []
        return self.params_by_preset.get(self.selected_key, [])


@dataclass
class AppState:
    """Shared runtime aggregate of the live domain objects (owned by the bridge)."""

    instruments: list[BaseInstrument] = field(default_factory=list)
    optocontrols: list[BaseOptoControl] = field(default_factory=list)
    displays: list["BaseDisplay"] = field(default_factory=list)
    preset: PresetState = field(default_factory=PresetState)
