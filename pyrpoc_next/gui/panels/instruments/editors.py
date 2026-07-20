"""GUI editors for instrument config, matched to an instrument by its key.

The new instruments are Qt-free and hold little config (most acquisition settings
moved to the routine), so their editors live here and stay small. An instrument
with no extra config has no entry — the manager falls back to the summary + Test
Connection body alone.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

from pyrpoc_next.instruments.base import Instrument
from pyrpoc_next.structs.keys import InstrumentKey

InstrumentEditor = Callable[[Instrument, Callable[[], None]], QWidget]

instrument_editors: dict[InstrumentKey, InstrumentEditor] = {}


def register(key: InstrumentKey):
    """Decorator: register an editor factory for one instrument key."""

    def decorate(factory: InstrumentEditor) -> InstrumentEditor:
        instrument_editors[key] = factory
        return factory

    return decorate


def editor_for(instrument: Instrument, on_change: Callable[[], None]) -> QWidget | None:
    """Build the config editor for this instrument, or None if it has no config."""
    factory = instrument_editors.get(instrument.key)
    return factory(instrument, on_change) if factory is not None else None


@register(InstrumentKey.ni_daq)
def ni_daq_editor(instrument: Instrument, on_change: Callable[[], None]) -> QWidget:
    """NI-DAQ has one config field: the device name (e.g. 'Dev1')."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel("Device name:"))
    field = QLineEdit(instrument.device_name)  # type: ignore[attr-defined]

    def apply() -> None:
        instrument.device_name = field.text().strip() or "Dev1"  # type: ignore[attr-defined]
        on_change()

    field.editingFinished.connect(apply)
    layout.addWidget(field, 1)
    return row
