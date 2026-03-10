from pyrpoc.backend_utils.registry import Registry
from .base_display import BaseDisplay


class DisplayRegistry(Registry):
    def __init__(self):
        super().__init__(name="DisplayRegistry", base_class=BaseDisplay)


display_registry = DisplayRegistry()
