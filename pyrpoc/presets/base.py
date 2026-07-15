from __future__ import annotations

from typing import Any

from pyrpoc.acquisition.setup import Setup
from pyrpoc.acquisition.source import CommandSource
from pyrpoc.utils.registry import Registry
from pyrpoc.structs.acquired_data import DataKind
from pyrpoc.structs.parameters import ParameterGroups


class Preset:
    """A named composition that builds the acquisition source + setup for a run.

    Class attributes declare what the preset needs and produces (used for UI
    filtering and requirement checks); ``build_source_and_setup`` constructs the
    concrete command source (later wrapped with optocontrol decorators by the
    bridge) and the once-per-run setup.
    """

    key: str = "base"
    display_name: str = "Base"
    parameter_groups: ParameterGroups = {}
    required_instruments: list[type] = []
    allowed_optocontrols: list[type] = []
    allowed_displays: list[str] = []
    emitted_kinds: list[DataKind] = []

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "preset_key": cls.key,
            "display_name": cls.display_name,
            "parameters": cls.parameter_groups,
            "required_instruments": cls.required_instruments,
            "allowed_optocontrols": cls.allowed_optocontrols,
            "emitted_kinds": cls.emitted_kinds,
            "allowed_displays": cls.allowed_displays,
        }

    def build_source_and_setup(
        self,
        *,
        params: dict[str, Any],
        instruments: dict[type, Any],
        frame_limit: int | None,
    ) -> tuple[CommandSource, Setup]:
        raise NotImplementedError


class PresetRegistry(Registry):
    def __init__(self):
        super().__init__(name="PresetRegistry", base_class=Preset)


preset_registry = PresetRegistry()
