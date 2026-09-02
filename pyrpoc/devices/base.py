"""What every device has: identity, configuration, and maybe a connection.

Separating identity from connection is what lets the galvo have a panel and be
persisted without pretending it opens a port -- it is mirrors moved by voltages
on someone else's AO channels, so it is ``backed_by`` the DAQ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pyrpoc.core import params as P

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from PyQt6.QtWidgets import QWidget


def make_instance_id(prefix: str) -> str:
    token = (prefix or "device").strip().lower()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in token)
    return f"{safe or 'device'}-{uuid4().hex[:12]}"


class Device:
    """One addressable piece of the instrument."""

    display_name: str = "Device"
    registry_key: str = "device"

    #: True when this device holds a resource that can be opened and verified.
    owns_connection: bool = False

    #: Set when this device has no connection of its own. Claims propagate up
    #: this link, so claiming the galvo claims its DAQ.
    backed_by: type["Device"] | None = None

    #: A ``core.params`` group describing this device's wiring and calibration.
    config_cls: type | None = None

    def __init__(self, instance_id: str | None = None, user_label: str | None = None):
        self.instance_id = instance_id or make_instance_id(self.registry_key)
        self.user_label = user_label
        self.last_error: str | None = None
        self.last_test_ok: bool | None = None
        self.config = self.config_cls() if self.config_cls is not None else None

    # -- identity ---------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self.user_label or self.display_name

    def summary(self) -> str:
        """One short line for the collapsed card in the devices panel."""
        return ""

    # -- connection -------------------------------------------------------- #

    def test_connection(self) -> bool:
        """Verify the device is reachable. Only meaningful when owns_connection.

        Records the result so the panel can show it after a restore.
        """
        if not self.owns_connection:
            return True
        try:
            ok = bool(self.check_reachable())
            self.last_error = None
        except Exception as exc:
            ok = False
            self.last_error = str(exc)
        self.last_test_ok = ok
        return ok

    def check_reachable(self) -> bool:
        """Subclass hook: raise or return False when the device is not there."""
        return True

    # -- panel ------------------------------------------------------------- #

    def panel(self, parent: "QWidget | None" = None, on_change=None) -> "QWidget | None":
        """Device-specific controls, beneath the form generated from ``config``.

        Returns None when the generated form is the whole panel. Subclasses
        import their widget *inside* this method: a module-scope Qt import would
        make ``pyrpoc.devices`` unimportable headless and fail
        ``tests/test_headless.py``.
        """
        del parent, on_change
        return None

    # -- persistence ------------------------------------------------------- #

    def export_state(self) -> dict[str, Any]:
        return {
            "config": P.to_dict(self.config) if self.config is not None else {},
            "last_test_ok": self.last_test_ok,
        }

    def import_state(self, raw: dict[str, Any]) -> None:
        if not isinstance(raw, dict):
            return
        if self.config_cls is not None:
            self.config = P.from_dict(self.config_cls, raw.get("config") or {})
        value = raw.get("last_test_ok")
        self.last_test_ok = value if isinstance(value, bool) else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.instance_id}>"
