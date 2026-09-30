"""Things you add more of. Each plugin subclasses a base class from structs
and registers itself; it imports only structs and qt_components (programs may
also use devices). Importing this package registers every plugin.

    devices/       hardware
    programs/      experiments, with their parameter groups and runners
    data_panels/   displays of one open dataset
"""

from __future__ import annotations

from .data_panels import data_panel_registry
from .devices import device_registry
from .programs import program_registry

__all__ = ["data_panel_registry", "device_registry", "program_registry"]
