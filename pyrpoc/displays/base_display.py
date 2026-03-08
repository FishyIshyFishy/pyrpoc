from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtWidgets import QWidget

from pyrpoc.backend_utils.contracts import ParameterGroups
from pyrpoc.backend_utils.data import BaseData
from pyrpoc.backend_utils.parameter_utils import validate_parameter_groups
from pyrpoc.rpoc.types import RPOCImageInput

if TYPE_CHECKING:
    from pyrpoc.domain.app_state import ParameterValue


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
        # Runtime metadata used by DisplayService/session persistence.
        self.attached: bool = True
        self.docked_visible: bool = True
        self.config_values: list[ParameterValue] = []
        self.user_label: str | None = None
        self.last_error: str | None = None

    @property
    def type_key(self) -> str:
        """Registry key used by session restore and inventory rows."""
        return self.DISPLAY_KEY

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

    def export_rpoc_input(self) -> RPOCImageInput | None:
        return None
