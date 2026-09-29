"""A widget that lives in the dock area: identity and a title."""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QWidget

from .device import make_instance_id


class Panel(QWidget):
    """A widget that lives in the dock area."""

    display_name: str = "Panel"
    registry_key: str = "panel"

    def __init__(self) -> None:
        super().__init__()
        self.instance_id = make_instance_id(self.registry_key)
        self.user_label: str | None = None

    @property
    def type_key(self) -> str:
        return self.registry_key

    @property
    def title(self) -> str:
        return self.user_label or self.display_name

    def export_persistence_state(self) -> dict[str, Any]:
        return {}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        del state
