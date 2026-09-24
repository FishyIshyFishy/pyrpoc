from __future__ import annotations

from pyrpoc.core.registry import Registry

from .base import Panel

panel_registry: Registry[Panel] = Registry("PanelRegistry", Panel)
