"""What every device has: identity, configuration, and maybe a connection.

Identity is separate from connection so a device with no port of its own, like
the galvo, can still be configured and persist; it is ``backed_by`` the DAQ.
A device with live controls has a ``DevicePanel``, registered separately in
``data_panels.py`` so this file stays Qt-free.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pyrpoc.structs.registry import Registry

from . import params as P


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

    def __init__(self, instance_id: str | None = None):
        self.instance_id = instance_id or make_instance_id(self.registry_key)
        # Which of its type this is, set by the inventory to tell same-kind
        # devices apart.
        self.number = 1
        self.last_error: str | None = None
        self.last_test_ok: bool | None = None
        self.config = self.config_cls()

    @property
    def name(self) -> str:
        return self.display_name if self.number == 1 else f"{self.display_name} {self.number}"

    def connection_info(self) -> str:
        """Where the device is reached, like a port or serial number, for the
        inventory table. Whether it is connected is shown separately."""
        return ""

    def test_connection(self) -> bool:
        """Verify the device is reachable, recording the result so the inventory
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
