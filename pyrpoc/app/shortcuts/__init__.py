"""Keyboard shortcuts: commands the window can run from a key, and the dialog that rebinds them."""

from __future__ import annotations

from .commands import Command, ShortcutController
from .dialog import ShortcutsDialog

__all__ = ["Command", "ShortcutController", "ShortcutsDialog"]
