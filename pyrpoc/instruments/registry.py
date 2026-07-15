from __future__ import annotations

from pyrpoc.utils.registry import Registry
from .base import BaseInstrument


class InstrumentRegistry(Registry):
    def __init__(self):
        super().__init__(name="InstrumentRegistry", base_class=BaseInstrument)


instrument_registry = InstrumentRegistry()
