from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pyrpoc.backend_utils.contracts import ParameterGroups
from pyrpoc.backend_utils.data import BaseData
from pyrpoc.backend_utils.parameter_utils import validate_parameter_groups
from pyrpoc.instruments.base_instrument import BaseInstrument


class BaseModality(ABC):
    MODALITY_KEY: str = "base_modality"
    DISPLAY_NAME: str = "Base Modality"
    PARAMETERS: ParameterGroups = {}
    REQUIRED_INSTRUMENTS: list[type[BaseInstrument]] = []
    OPTIONAL_INSTRUMENTS: list[type[BaseInstrument]] = []
    OUTPUT_DATA_TYPE: type[BaseData] = BaseData
    ALLOWED_DISPLAYS: list[str] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        required = getattr(cls, "REQUIRED_INSTRUMENTS", [])
        optional = getattr(cls, "OPTIONAL_INSTRUMENTS", [])

        if not isinstance(required, list):
            raise TypeError("REQUIRED_INSTRUMENTS must be a list")
        if not isinstance(optional, list):
            raise TypeError("OPTIONAL_INSTRUMENTS must be a list")

        for instrument_cls in [*required, *optional]:
            if not isinstance(instrument_cls, type) or not issubclass(instrument_cls, BaseInstrument):
                raise TypeError(
                    f"{cls.__name__} instrument requirements must contain BaseInstrument subclasses"
                )

        validate_parameter_groups(getattr(cls, "PARAMETERS", {}))

        output_type = getattr(cls, "OUTPUT_DATA_TYPE", BaseData)
        if not isinstance(output_type, type) or not issubclass(output_type, BaseData):
            raise TypeError("OUTPUT_DATA_TYPE must be a BaseData subclass")

    def __init__(self):
        self._running = False
        self._configured = False
        self._params: dict[str, Any] = {}
        self._instruments: dict[type[BaseInstrument], BaseInstrument] = {}

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "modality_key": cls.MODALITY_KEY,
            "display_name": cls.DISPLAY_NAME,
            "parameters": cls.PARAMETERS,
            "required_instruments": cls.REQUIRED_INSTRUMENTS,
            "optional_instruments": cls.OPTIONAL_INSTRUMENTS,
            "output_data_type": cls.OUTPUT_DATA_TYPE,
            "allowed_displays": cls.ALLOWED_DISPLAYS,
        }

    @abstractmethod
    def configure(
        self,
        params: dict[str, Any],
        instruments: dict[type[BaseInstrument], BaseInstrument],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def acquire_once(self) -> BaseData:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
