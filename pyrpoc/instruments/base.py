from __future__ import annotations

from typing import Any

from pyrpoc.utils.state_helpers import export_object_state, import_object_state, make_instance_id


class BaseInstrument:
    """Hardware-control logic for one instrument instance (Qt-free).

    The editor widget is provided separately by ``gui``, matched to this class
    by ``instrument_key`` through a widget registry — this class never builds Qt.
    """

    instrument_key: str = "base_instrument"
    display_name: str = "Base Instrument"
    persistence_fields: tuple[str, ...] | None = None
    persistence_exclude_fields: tuple[str, ...] = (
        "alias",
        "connected",
        "instance_id",
        "last_error",
        "user_label",
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

    def prepare_for_acquisition(self) -> tuple[Any, ...]:
        """Return a compact acquisition payload (stable identifiers)."""
        return (self.alias,)

    def get_collapsed_summary(self) -> str:
        """Return short text shown next to the instrument name in collapsed cards."""
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
