"""Modifier configuration dataclasses. Importing registers each modifier."""

from __future__ import annotations

from pyrpoc_next.acquisition.modifiers.base import Modifier, modifier_registry
from pyrpoc_next.acquisition.modifiers.mask import MaskModifier

__all__ = ["Modifier", "modifier_registry", "MaskModifier"]
