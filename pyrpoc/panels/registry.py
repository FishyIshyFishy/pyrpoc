from __future__ import annotations

from pyrpoc.structs.registries import Registry

from .base import Panel

panel_registry: Registry[Panel] = Registry("PanelRegistry", Panel)
