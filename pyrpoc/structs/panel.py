"""A widget that lives in the dock area: identity and a title, no more.

All seven panels inherit this -- devices and the data library exactly as much
as image_2d does. What used to live here besides identity was a source
picker: a combo box naming which open dataset to show. That is not a
property of "being a panel", it is a property of showing a dataset that
outlives its renderer, so it moved to ``components/source_picker.py`` as a
widget a panel adds to its own layout, not something every panel is forced to
carry. That is what keeps devices and acquisition from needing a "source"
with nothing to put in it.

The persistence hooks are the other half of what "thin" buys: a future
docking/layout system has one place to call, on every panel, without any of
them having had to know that was coming.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from PyQt6.QtWidgets import QWidget

from .registries import Registry


def make_instance_id(prefix: str) -> str:
    token = (prefix or "panel").strip().lower()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in token)
    return f"{safe or 'panel'}-{uuid4().hex[:12]}"


class Panel(QWidget):
    """A widget that lives in the dock area. Identity and a title, no more."""

    display_name: str = "Panel"
    registry_key: str = "panel"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.instance_id = make_instance_id(self.registry_key)
        self.user_label: str | None = None
        self.last_error: str | None = None

    @property
    def type_key(self) -> str:
        return self.registry_key

    @property
    def title(self) -> str:
        return self.user_label or self.display_name

    # -- persistence ---------------------------------------------------------- #

    def export_persistence_state(self) -> dict[str, Any]:
        return {}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        del state


panel_registry: Registry[Panel] = Registry("PanelRegistry", Panel)
