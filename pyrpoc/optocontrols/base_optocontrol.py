from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PyQt6.QtWidgets import QWidget


class BaseOptoControl(ABC):
    OPTOCONTROL_KEY: str = "base_optocontrol"
    DISPLAY_NAME: str = "Base Opto-Control"

    def __init__(self, alias: str | None = None):
        self.alias = alias or self.OPTOCONTROL_KEY

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "optocontrol_key": cls.OPTOCONTROL_KEY,
            "display_name": cls.DISPLAY_NAME,
        }

    @abstractmethod
    def get_widget(self, parent: QWidget | None = None) -> QWidget:
        raise NotImplementedError

    @abstractmethod
    def prepare_data_for_acquisition(self) -> dict[str, Any]:
        raise NotImplementedError
