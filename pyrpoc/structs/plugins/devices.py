"""What every device has: identity, configuration, and maybe a connection.

Identity is separate from connection so a device with no port of its own, like
the galvo, can still have a panel and persist; it is ``backed_by`` the DAQ.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pyrpoc.structs.registry import Registry

from . import params as P

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from PyQt6.QtWidgets import QWidget


def make_instance_id(prefix: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in prefix.lower())
    return f"{safe}-{uuid4().hex[:12]}"


class DeviceError(Exception):
    """A device could not be reached or configured."""


class MissingDevice(DeviceError):
    """A program needs devices that are not in the inventory."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__("missing required devices: " + ", ".join(self.missing))


class Device:
    """One addressable piece of the instrument."""

    display_name: str = "Device"
    registry_key: str = "device"

    # True when this device holds a resource that can be opened and verified.
    owns_connection: bool = False

    # True when this device keeps a connection open between runs: the app
    # opens it at launch and closes it on exit.
    holds_session: bool = False

    # Set when this device has no connection of its own. Claims propagate up
    # this link, so claiming the galvo claims its DAQ.
    backed_by: type[Device] | None = None

    # The block describing this device's wiring and calibration.
    config_cls: type[P.Group]

    def __init__(self, instance_id: str | None = None, user_label: str | None = None):
        self.instance_id = instance_id or make_instance_id(self.registry_key)
        self.user_label = user_label
        self.last_error: str | None = None
        self.last_test_ok: bool | None = None
        self.config = self.config_cls()

    @property
    def name(self) -> str:
        return self.user_label or self.display_name

    def summary(self) -> str:
        """One short line for the collapsed card in the devices panel."""
        return ""

    def test_connection(self) -> bool:
        """Verify the device is reachable, recording the result so the panel
        can show it after a restore."""
        if not self.owns_connection:
            return True
        # A hardware boundary: the SDK raises whatever it raises.
        try:
            ok = self.check_reachable()
            self.last_error = None
        except Exception as exc:
            ok = False
            self.last_error = str(exc)
        self.last_test_ok = ok
        return ok

    def check_reachable(self) -> bool:
        """Subclass hook: raise or return False when the device is not there."""
        return True

    @property
    def session_open(self) -> bool:
        return False

    def open_session(self) -> None:
        """Subclass hook for a ``holds_session`` device."""
        raise NotImplementedError

    def close_session(self) -> None:
        """Subclass hook for a ``holds_session`` device."""
        raise NotImplementedError

    def try_open_session(self) -> bool:
        """Open the session, recording why it failed instead of raising, so a
        device that is switched off cannot stop the app from starting."""
        # A hardware boundary: the SDK raises whatever it raises.
        try:
            self.open_session()
        except Exception as exc:
            self.last_error = str(exc)
            return False
        self.last_error = None
        return True

    def panel(self, parent: QWidget, on_change: Callable[[], None]) -> QWidget | None:
        """Device-specific controls beneath the generated config form, or None.
        Subclasses import Qt inside this method so devices/ stays importable
        without a display."""
        del parent, on_change
        return None

    def export_state(self) -> dict[str, Any]:
        return {"config": P.encode_block(self.config), "last_test_ok": self.last_test_ok}

    def import_state(self, raw: dict[str, Any]) -> None:
        """Restore from session JSON, which is a boundary: a malformed entry
        leaves that part at its default."""
        config = raw.get("config")
        if isinstance(config, dict):
            self.config = P.decode_block(self.config_cls, config)
        value = raw.get("last_test_ok")
        self.last_test_ok = value if isinstance(value, bool) else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.instance_id}>"


device_registry: Registry[Device] = Registry("DeviceRegistry", Device)
