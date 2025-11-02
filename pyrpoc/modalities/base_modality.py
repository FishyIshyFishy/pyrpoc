from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from pyrpoc.backend_utils.data import BaseData
from pyrpoc.backend_utils.parameter import BaseParameter
from pyrpoc.instruments import BaseInstrument
from pyrpoc.laser_modulations.base_laser_mod import BaseLaserModulation

class BaseModality(ABC):
    '''
    description:
        high level Modality that the GUI sees and communicates with
    '''
    required_parameters: List[Type[BaseParameter]] = [] 
    required_instruments: List[Type[BaseInstrument]] = []
    allowed_modulations: List[Type[BaseLaserModulation]] = []
    emission_data_type: Type[BaseData] = BaseData

    def __init__(self, name: str, **kwargs):
        self.name = name
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.params = kwargs  # keep a copy for metadata / saving, but this is not to be read from in general

    @classmethod
    def get_contract(cls) -> dict:
        return {
            'parameters': cls.required_parameters,
            'instruments': cls.required_instruments,
            'modulations': cls.allowed_modulations,
            'emission_data_type': cls.emission_data_type,
        }

    @classmethod
    def get_required_parameters(cls) -> List[Type[BaseParameter]]:
        return cls.required_parameters

    @classmethod
    def get_required_instruments(cls) -> List[Type[BaseInstrument]]:
        return cls.required_instruments
    
    @classmethod
    def get_allowed_modulations(cls) -> List[Type[BaseLaserModulation]]:
        return cls.allowed_modulations

    @classmethod
    def get_emission_data_type(cls) -> Type[BaseData]:
        return cls.emission_data_type


    def start_acquisition(self, context):
        pass

    def stop_acquisition(self):
        pass

    def set_acquisition_params(self, params: dict[str, Any]):
        self.acquisition_params = params

    def set_acquisition_instruments(self, instruments: List[BaseInstrument]):
        '''
        description:
            Given a list of connected instrument instances, this function assigns
            the ones matching each required instrument type defined by the modality.

        args:
            instruments: list of BaseInstrument instances currently connected.
        '''
        self.acquisition_instruments: Dict[Type[BaseInstrument], BaseInstrument] = {}

        for req_cls in self.required_instruments:
            matched = None
            for inst in instruments:
                if isinstance(inst, req_cls):
                    matched = inst
                    break

            if matched is not None:
                self.acquisition_instruments[req_cls] = matched
            else:
                raise RuntimeError(f'[{self.name}] missing required instrument: {req_cls.__name__}')

    def set_acquisition_modulations(self, modulations: List[BaseLaserModulation]):
        pass



    @abstractmethod
    def perform_acquisition(self):
        pass