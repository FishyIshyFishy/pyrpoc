from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import QWidget


class BaseInstrument(ABC):
    INSTRUMENT_KEY: str = "base_instrument"
    DISPLAY_NAME: str = "Base Instrument"

    def __init__(self, alias: str | None = None):
        self.alias = alias or self.INSTRUMENT_KEY

    @property
    def type_key(self) -> str:
        """Registry key used by persistence and inventory rows to recreate this instance."""
        return self.alias

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "instrument_key": cls.INSTRUMENT_KEY,
            "display_name": cls.DISPLAY_NAME,
        }

    @classmethod
    def get_widget_contract(cls) -> dict[str, Any]:
        """Return only UI-related metadata for discovery-driven widgets.

        Call flow:
        - UI code queries `instrument_service.list_available()`
        - `Registry.describe()` returns `get_widget_contract()` in addition to `get_contract()`
        - Managers can use this to skip special-case imports while deciding how to render.
        """
        return {}

    @abstractmethod
    def get_widget(
        self,
        parent: QWidget | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> "BaseInstrumentWidget":
        """Create or reuse a concrete widget for this instance.

        Call flow:
        - user expands an instance card in InstrumentManager
        - instrument_mgr.handlers.on_expand_requested asks InstrumentService for the widget
        - InstrumentService delegates to this method to keep all concrete imports localized.
        """
        raise NotImplementedError

    def prepare_for_acquisition(self) -> tuple[Any, ...]:
        """Return a compact acquisition payload for future modality hooks.

        Call flow:
        - modality code may call this after requirements validation
        - default payload includes only stable identifiers to keep phase-1 minimal.
        """
        return (self.alias,)

    def get_collapsed_summary(self) -> str:
        """Return short text shown next to instrument name in collapsed manager cards."""
        return ""


class BaseInstrumentWidget(QWidget):
    """Base Qt widget for a single instrument instance.

    Call flow:
    - user clicks Expand in InstrumentManager card
    - instrument_mgr.handlers.on_expand_requested -> InstrumentService.get_widget
    - that call returns a `BaseInstrumentWidget` from concrete `BaseInstrument.get_widget`
    """

    def __init__(self, instrument: BaseInstrument, on_change: Callable[[], None] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.instrument = instrument
        self.on_change = on_change

    def refresh_from_model(self) -> None:
        """Load current instrument-level model state after restore/rebuild."""
        return None

    def _request_model_persist(self) -> None:
        if self.on_change is not None:
            self.on_change()
