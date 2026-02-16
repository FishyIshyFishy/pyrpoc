from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QWidget

from pyrpoc.backend_utils.contracts import ParameterGroups
from pyrpoc.backend_utils.data import BaseData
from pyrpoc.backend_utils.parameter_utils import validate_parameter_groups


class BaseDisplay(QWidget):
    DISPLAY_KEY: str = "base_display"
    DISPLAY_NAME: str = "Base Display"
    ACCEPTED_DATA_TYPES: list[type[BaseData]] = [BaseData]
    DISPLAY_PARAMETERS: ParameterGroups = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        accepted = getattr(cls, "ACCEPTED_DATA_TYPES", [])
        if not isinstance(accepted, list):
            raise TypeError("ACCEPTED_DATA_TYPES must be a list")
        for data_type in accepted:
            if not isinstance(data_type, type) or not issubclass(data_type, BaseData):
                raise TypeError("ACCEPTED_DATA_TYPES must contain BaseData subclasses")

        validate_parameter_groups(getattr(cls, "DISPLAY_PARAMETERS", {}))

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "display_key": cls.DISPLAY_KEY,
            "display_name": cls.DISPLAY_NAME,
            "accepted_data_types": cls.ACCEPTED_DATA_TYPES,
            "display_parameters": cls.DISPLAY_PARAMETERS,
        }

    def configure(self, params: dict[str, Any]) -> None:
        raise NotImplementedError

    def render(self, data: BaseData) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError
