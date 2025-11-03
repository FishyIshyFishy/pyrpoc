from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from pyrpoc.utils.data import BaseData
from pyrpoc.utils.parameter import BaseParameter
from pyrpoc.utils.context import AcquisitionContext
from pyrpoc.instruments import BaseInstrument
from pyrpoc.laser_modulations.base_laser_mod import BaseLaserModulation


class BaseModality(ABC):
    '''
    description:
        Abstract base class for all modalities.  
        Each modality defines required parameters, required instruments,
        allowed modulations, and the data type it emits.
    '''

    required_parameters: list[BaseParameter] = []
    required_instruments: list[type[BaseInstrument]] = []
    allowed_modulations: list[type[BaseLaserModulation]] = []
    emission_data_type: type[BaseData]

    def __init__(self, context: AcquisitionContext):
        self.context = context
        self.acquisition_params: dict[str, Any] = {}
        self.acquisition_instruments: dict[type[BaseInstrument], BaseInstrument] = {}
        self.acquisition_modulations: list[BaseLaserModulation] = []

    # -------------------------------------------------------------------------
    # --- Acquisition Lifecycle ----------------------------------------------
    # -------------------------------------------------------------------------

    def start_acquisition(self, context: AcquisitionContext):
        '''
        description:
            High-level acquisition entry point.
            Should not depend on GUI code — only on the context object.
        '''
        self.set_params(context.params)
        self.set_instruments(context.instruments)
        self.set_modulations(context.mods)
        result = self.perform_acquisition()
        self.finish_acquisition(result)

    def set_params(self, parameters: list[BaseParameter]):
        '''
        description:
            Converts BaseParameter objects into plain attributes
            and stores them for internal use.
        '''
        for param in parameters:
            setattr(self, param.name, param.value)

    def set_instruments(self, instruments:list[BaseInstrument]):
        '''
        description:
            match connected instrument instances to required types.
            note that if multiple instruments of a given type are connected, 
            the matched instrument is only the first one to appear in the list

            this may be an unpredictable order, depending on GUI config
            it will be best to connect only a single instrument of each type
        '''
        self.acquisition_instruments = {}
        for req_cls in self.required_instruments:
            matched = None
            for inst in instruments:
                if isinstance(inst, req_cls):
                    matched = inst
                    break

            if matched is None:
                raise RuntimeError(f'Missing required instrument: {req_cls.__name__}')
            self.acquisition_instruments[req_cls] = matched

    def set_modulations(self, modulations: list[BaseLaserModulation]):
        '''
        description:
            Assigns available laser modulations to this modality.
            Validation ensures only allowed modulation types are included.
        '''
        self.acquisition_modulations = []
        for mod in modulations:
            if type(mod) not in self.allowed_modulations:
                raise RuntimeError(
                    f'Modulation {type(mod).__name__} not allowed for this modality.'
                )
            self.acquisition_modulations.append(mod)

    @abstractmethod
    def perform_acquisition(self) -> BaseData:
        '''
        description:
            Core acquisition routine (to be implemented by subclass).
            Should return data of type `emission_data_type`.
        '''
        pass

    def finish_acquisition(self, result: BaseData | None):
        '''
        description:
            Finalization step after acquisition completes.
            Handles cleanup, logging, or data packaging.
        '''
        if result is not None:
            print(f'Acquisition finished, got data of type {type(result).__name__}')
        else:
            print(f'Acquisition finished with no data.')






    @classmethod
    def get_contract(cls) -> dict:
        return {
            'parameters': cls.required_parameters,
            'instruments': cls.required_instruments,
            'modulations': cls.allowed_modulations,
            'emission_data_type': cls.emission_data_type,
        }

    @classmethod
    def get_required_parameters(cls) -> list[BaseParameter]:
        return cls.required_parameters

    @classmethod
    def get_required_instruments(cls) -> list[type[BaseInstrument]]:
        return cls.required_instruments

    @classmethod
    def get_allowed_modulations(cls) -> list[type[BaseLaserModulation]]:
        return cls.allowed_modulations

    @classmethod
    def get_emission_data_type(cls) -> type[BaseData]:
        return cls.emission_data_type

