from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.instruments import BaseInstrument
from pyrpoc.laser_modulations.base_laser_mod import BaseLaserModulation


class BaseModality(ABC):
    REQUIRED_PARAMETERS: Dict[str, Dict[str, Dict[str, Any]]] = {}
    REQUIRED_INSTRUMENTS: List[Type[BaseInstrument]] = []
    ALLOWED_DISPLAYS: List[str] = []
    ALLOWED_MODULATONS: List[Type[BaseLaserModulation]] = []
    DATA_TYPE: Type[BaseData] = BaseData

    def __init__(self, name: str, **kwargs):
        self.name = name
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.params = kwargs  # keep a copy for metadata / saving, but this is not to be read from in general

    @classmethod
    def get_contract(cls) -> dict:
        return {
            'parameters': cls.REQUIRED_PARAMETERS,
            'instruments': cls.REQUIRED_INSTRUMENTS,
            'displays': cls.ALLOWED_DISPLAYS,
            'data_type': cls.DATA_TYPE,
        }

    @classmethod
    def get_required_parameters(cls) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return cls.REQUIRED_PARAMETERS

    @classmethod
    def get_required_instruments(cls) -> List[Type[BaseInstrument]]:
        return cls.REQUIRED_INSTRUMENTS

    @classmethod
    def get_allowed_displays(cls) -> List[str]:
        return cls.ALLOWED_DISPLAYS

    @classmethod
    def get_data_type(cls) -> Type[BaseData]:
        return cls.DATA_TYPE

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def emit_data_object(self, data: BaseData):
        if not isinstance(data, self.DATA_TYPE):
            raise TypeError(
                f'emit_data_object expected {self.DATA_TYPE.__name__}, '
                f'got {type(data).__name__}'
            )
        print(f'[{self.name}] emitted {data}')


    def get_metadata(self) -> dict:
        return {
            'name': self.name,
            'parameters': self.params,
            'contract': self.get_contract(),
        }
