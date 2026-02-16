from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pyrpoc.backend_utils.contracts import Action, ParameterGroups
from pyrpoc.backend_utils.parameter_utils import validate_action_list, validate_parameter_groups


class BaseOptoControl(ABC):
    OPTOCONTROL_KEY: str = "base_optocontrol"
    DISPLAY_NAME: str = "Base Opto-Control"
    CONFIG_PARAMETERS: ParameterGroups = {}
    ACTIONS: list[Action] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validate_parameter_groups(getattr(cls, "CONFIG_PARAMETERS", {}))
        validate_action_list(getattr(cls, "ACTIONS", []))

    def __init__(self, alias: str | None = None):
        self.alias = alias or self.OPTOCONTROL_KEY
        self._connected = False

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "optocontrol_key": cls.OPTOCONTROL_KEY,
            "display_name": cls.DISPLAY_NAME,
            "config_parameters": cls.CONFIG_PARAMETERS,
            "actions": cls.ACTIONS,
        }

    @abstractmethod
    def connect(self, config: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        raise NotImplementedError

    def execute_action(self, method_name: str, args: dict[str, Any]) -> None:
        method = getattr(self, method_name, None)
        if method is None:
            raise AttributeError(f"{self.__class__.__name__} does not implement action method '{method_name}'")
        if not callable(method):
            raise TypeError(f"attribute '{method_name}' on {self.__class__.__name__} is not callable")
        method(args)
