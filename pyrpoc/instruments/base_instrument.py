from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import QWidget

from pyrpoc.backend_utils.state_helpers import export_object_state, import_object_state, make_instance_id


class BaseInstrument(ABC):
    instrument_key: str = "base_instrument"
    display_name: str = "Base Instrument"
    persistence_fields: tuple[str, ...] | None = None
    persistence_exclude_fields: tuple[str, ...] = (
        "alias",
        "connected",
        "instance_id",
        "last_error",
        "user_label",
        "widget",
    )

    def __init__(
        self,
        alias: str | None = None,
        *,
        instance_id: str | None = None,
        user_label: str | None = None,
        connected: bool = False,
    ):
        self.alias = alias or self.instrument_key
        self.instance_id = instance_id or make_instance_id(self.alias)
        self.user_label = user_label
        self.connected = bool(connected)
        self.last_error: str | None = None

    @property
    def type_key(self) -> str:
        """Registry key used by persistence and inventory rows to recreate this instance."""
        return self.alias

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "instrument_key": cls.instrument_key,
            "display_name": cls.display_name,
        }

    @classmethod
    def get_widget_contract(cls) -> dict[str, Any]:
        """Return UI-only metadata used by discovery-driven manager widgets."""
        return {}

    @abstractmethod
    def get_widget(
        self,
        parent: QWidget | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> "BaseInstrumentWidget":
        """Create or reuse the concrete editor widget for this instance."""
        raise NotImplementedError

    def prepare_for_acquisition(self) -> tuple[Any, ...]:
        """Return a compact acquisition payload (stable identifiers) for modality hooks."""
        return (self.alias,)

    def get_collapsed_summary(self) -> str:
        """Return short text shown next to instrument name in collapsed manager cards."""
        return ""

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

    def connect(self) -> bool:
        return True


class BaseInstrumentWidget(QWidget):
    """Base Qt widget for a single instrument instance."""

    def __init__(self, instrument: BaseInstrument, on_change: Callable[[], None] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.instrument = instrument
        self.on_change = on_change

    def refresh_from_model(self) -> None:
        """Load current instrument-level model state after restore/rebuild."""
        return None

    def request_model_persist(self) -> None:
        if self.on_change is not None:
            self.on_change()
