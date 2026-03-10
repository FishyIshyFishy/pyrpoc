from pyrpoc.backend_utils.registry import Registry
from .base_instrument import BaseInstrument


class InstrumentRegistry(Registry):
    def __init__(self):
        super().__init__(name="InstrumentRegistry", base_class=BaseInstrument)


instrument_registry = InstrumentRegistry()
