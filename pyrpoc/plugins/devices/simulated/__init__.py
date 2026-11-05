"""Simulated devices, which acquire from a simulated dish. Only ``pyrpoc --simulated`` imports
this package."""

from __future__ import annotations

from pyrpoc.core.device import device_registry
from pyrpoc.core.panels import device_panel_registry

from .dish import SimulatedDish
from .panel import SimulatedDishPanel
from .scanner import SimulatedScanner

__all__ = ["SimulatedDish", "SimulatedScanner", "register"]


def register() -> None:
    """Add the simulated devices to the device list."""
    # Registered here rather than where each is defined, so they exist only when this is called.
    device_registry.add("simulated_dish", SimulatedDish)
    device_registry.add("simulated_scanner", SimulatedScanner)
    device_panel_registry.add("simulated_dish", SimulatedDishPanel)
