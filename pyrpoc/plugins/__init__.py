"""Things you add more of: devices, program inputs and outputs, programs and panels. Importing
this package registers every one."""

from __future__ import annotations

from . import program_io
from .devices import device_registry
from .panels import data_panel_registry
from .programs import program_registry

__all__ = ["data_panel_registry", "device_registry", "program_io", "program_registry"]
