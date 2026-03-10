from __future__ import annotations

from typing import Any


class Registry:
    def __init__(self, name: str, base_class: type):
        self.name = name
        self.base_class = base_class
        self.entries: dict[str, type] = {}

    def register(self, key: str):
        def decorator(cls: type):
            if not issubclass(cls, self.base_class):
                raise TypeError(f"{cls.__name__} must inherit from {self.base_class.__name__}")
            if key in self.entries:
                raise KeyError(f"{key!r} is already registered in {self.name}")
            self.entries[key] = cls
            return cls

        return decorator

    def list_keys(self) -> list[str]:
        return sorted(self.entries.keys())

    def get_registered(self) -> list[str]:
        return self.list_keys()

    def get_class(self, key: str) -> type:
        if key not in self.entries:
            raise KeyError(f"{key!r} is not registered in {self.name}")
        return self.entries[key]

    def create(self, key: str, **kwargs: Any) -> Any:
        cls = self.get_class(key)
        return cls(**kwargs)

    def describe(self, key: str) -> dict[str, Any]:
        cls = self.get_class(key)
        return {
            "key": key,
            "class_name": cls.__name__,
            "display_name": getattr(cls, "DISPLAY_NAME", cls.__name__),
            "parameters": getattr(cls, "PARAMETERS", {}),
            "config_parameters": getattr(cls, "CONFIG_PARAMETERS", {}),
            "display_parameters": getattr(cls, "DISPLAY_PARAMETERS", {}),
            "actions": getattr(cls, "ACTIONS", []),
            "contract": cls.get_contract() if hasattr(cls, "get_contract") else {},
        }

    def describe_all(self) -> list[dict[str, Any]]:
        return [self.describe(key) for key in self.list_keys()]
