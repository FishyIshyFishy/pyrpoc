from __future__ import annotations

from pyrpoc.utils.registry import Registry
from .base import BaseOptoControl


class OptoControlRegistry(Registry):
    def __init__(self):
        super().__init__(name="OptoControlRegistry", base_class=BaseOptoControl)


opto_control_registry = OptoControlRegistry()
