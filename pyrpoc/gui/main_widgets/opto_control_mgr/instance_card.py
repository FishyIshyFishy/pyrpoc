"""Re-export shim — InstanceCardWidget now lives in the shared module."""

from pyrpoc.gui.main_widgets.instance_card import BaseCardWidget, RemovableCardWidget

# InstanceCardWidget kept as an alias so that any remaining callsites that
# haven't been migrated to the new names continue to work.
InstanceCardWidget = RemovableCardWidget

__all__ = ["BaseCardWidget", "InstanceCardWidget", "RemovableCardWidget"]
