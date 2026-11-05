"""``Device``: one piece of the instrument, with its identity, its
configuration, and maybe a connection."""

from __future__ import annotations

from uuid import uuid4

from pyrpoc.core.registry import Registry

from . import device_config as C


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

    # The configuration describing this device's wiring and calibration.
    config_cls: type[C.DeviceConfig]

    def __init__(self, instance_id: str | None = None):
        self.instance_id = instance_id or make_instance_id(self.registry_key)
        self.config = self.config_cls()

    def connection_info(self) -> str:
        """Where the device is reached, like a port or serial number."""
        return ""

    def check_reachable(self) -> bool:
        """Raise or return False when the device is not there."""
        return True

    @property
    def session_open(self) -> bool:
        return False

    def open_session(self) -> None:
        """Open the connection a ``holds_session`` device keeps."""
        raise NotImplementedError

    def close_session(self) -> None:
        """Close the connection a ``holds_session`` device keeps."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.instance_id}>"


device_registry: Registry[Device] = Registry("DeviceRegistry", Device)
