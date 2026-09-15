from __future__ import annotations

from pyrpoc.core.registry import Registry

from .base import View

view_registry: Registry[View] = Registry("ViewRegistry", View)
