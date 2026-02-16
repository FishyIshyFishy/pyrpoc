from pyrpoc.backend_utils.registry import Registry
from .base_optocontrol import BaseOptoControl


class OptoControlRegistry(Registry):
    def __init__(self):
        super().__init__(name="OptoControlRegistry", base_class=BaseOptoControl)


opto_control_registry = OptoControlRegistry()
