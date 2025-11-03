import numpy as np
from dataclasses import dataclass
from typing import Any

from pyrpoc.utils.parameter import BaseParameter
from pyrpoc.instruments.base_instrument import BaseInstrument
from pyrpoc.laser_modulations.base_laser_mod import BaseLaserModulation
from pyrpoc.displays.base_display import BaseDisplay

@dataclass
class AcquisitionContext:
    params: list[BaseParameter]
    instruments: list[BaseInstrument]
    mods: list[BaseLaserModulation]
    display_type: BaseDisplay

    save: bool
    save_path: str | None = None