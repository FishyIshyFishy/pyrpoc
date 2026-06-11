from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PyQt6.QtWidgets import QWidget

from pyrpoc.backend_utils.contracts import ParameterGroups
from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.backend_utils.parameter_utils import validate_parameter_groups
from pyrpoc.backend_utils.state_helpers import export_object_state, import_object_state, make_instance_id
from pyrpoc.rpoc.types import RPOCImageInput

if TYPE_CHECKING:
    from pyrpoc.domain.app_state import ParameterValue
    from collections.abc import Callable


class BaseDisplay(QWidget):
    display_key: str = "base_display"
    display_name: str = "Base Display"
    accepted_kinds: list[DataKind] = []
    display_parameters: ParameterGroups = {}
    persistence_fields: tuple[str, ...] | None = None
    persistence_exclude_fields: tuple[str, ...] = (
        "attached",
        "config_values",
        "docked_visible",
        "instance_id",
        "last_error",
        "user_label",
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        accepted = getattr(cls, "accepted_kinds", [])
        if not isinstance(accepted, list):
            raise TypeError("accepted_kinds must be a list")

        validate_parameter_groups(getattr(cls, "display_parameters", {}))

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        # Runtime metadata used by DisplayService/session persistence.
        self.attached: bool = True
        self.docked_visible: bool = True
        self.config_values: list[ParameterValue] = []
        self.user_label: str | None = None
        self.instance_id: str = make_instance_id(self.display_key)
        self.last_error: str | None = None
        self._persist_callback: Callable[[], None] | None = None

    @property
    def type_key(self) -> str:
        """Registry key used by session restore and inventory rows."""
        return self.display_key

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "display_key": cls.display_key,
            "display_name": cls.display_name,
            "accepted_kinds": cls.accepted_kinds,
            "display_parameters": cls.display_parameters,
        }

    def configure(self, params: dict[str, Any]) -> None:
        raise NotImplementedError

    def render(self, acquired: AcquiredData) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def export_rpoc_input(self) -> RPOCImageInput | None:
        return None

    def get_normalized_data_3d(self) -> np.ndarray | None:
        """Return current display data as float32 [C,H,W] in [0,1], or None if unavailable."""
        return None

    def export_persistence_state(self) -> dict[str, Any]:
        return export_object_state(
            self,
            include_fields=self.persistence_fields,
            exclude_fields=self.persistence_exclude_fields,
        )

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        import_object_state(
            self,
            state,
            include_fields=self.persistence_fields,
            exclude_fields=self.persistence_exclude_fields,
        )

    def set_persist_callback(self, callback: "Callable[[], None] | None") -> None:
        self._persist_callback = callback

    def request_persist(self) -> None:
        if self._persist_callback is not None:
            self._persist_callback()
