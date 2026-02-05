import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional
from abc import ABC, abstractmethod

from pyrpoc.utils.parameters import BaseParameter
from pyrpoc.utils.datas import BaseData
from pyrpoc.utils.base_types import BaseInstrument, BaseLaserModulation, BaseDisplay


@dataclass
class BaseContext(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


@dataclass
class AcquisitionContext(BaseContext):
    parameters: dict[str, BaseParameter]
    instruments: list[BaseInstrument] = []
    laser_modulations: list[BaseLaserModulation] = []

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class DataContext(BaseContext):
    data: dict[str, BaseData] # name: data
    final: bool


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
