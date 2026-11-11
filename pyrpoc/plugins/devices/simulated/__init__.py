"""Simulated devices, which acquire from the simulated dish in ``utils/simulation``. Only
``pyrpoc --simulated`` imports this package, and importing it registers them."""

from __future__ import annotations

from .dish import SimulatedDish
from .panel import SimulatedDishPanel
from .scanner import SimulatedScanner

__all__ = ["SimulatedDish", "SimulatedDishPanel", "SimulatedScanner"]
