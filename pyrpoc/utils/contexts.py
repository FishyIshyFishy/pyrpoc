import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional
from abc import ABC, abstractmethod

from pyrpoc.utils.parameters import BaseParameter
from pyrpoc.utils.datas import BaseData
from pyrpoc.utils.base_types.base_instrument import BaseInstrument
from pyrpoc.laser_modulations.base_laser_mod import BaseLaserModulation
from pyrpoc.utils.base_types.base_display import BaseDisplay


@dataclass
class BaseContext(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


@dataclass
class AcquisitionContext(BaseContext):
    parameters: dict[str, BaseParameter] = field(default_factory=dict)
    instruments: list[BaseInstrument] = field(default_factory=list)
    laser_modulations: list[BaseLaserModulation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class DataContext(BaseContext):
    data: BaseData

    final: bool = False
    display_params: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

@dataclass
class ModalityContext(BaseContext):
    required_parameters: list[BaseParameter] = field(default_factory=list)
    required_instruments: list[type[BaseInstrument]] = field(default_factory=list)
    allowed_modulations: list[type[BaseLaserModulation]] = field(default_factory=list)
    default_display: Optional[type[BaseDisplay]] = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__
