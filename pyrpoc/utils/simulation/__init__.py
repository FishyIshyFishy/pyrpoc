"""A simulated dish of moving, bleaching cells under a drifting focus, with its own clock and
stage, for simulated devices to acquire from."""

from __future__ import annotations

from .dish import CHANNELS, Dish, DishSettings
from .optics import FocalSurface
from .stage import Axis

__all__ = ["CHANNELS", "Axis", "Dish", "DishSettings", "FocalSurface"]
