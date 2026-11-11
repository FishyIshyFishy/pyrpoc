"""``Device``: one piece of the instrument, with its identity, its configuration, and maybe a
connection; and ``DeviceConfig``, its settings, whose fields carry the rules their values
follow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from dataclasses import field as dc_field
from enum import Enum
from typing import Any
from uuid import uuid4

from pyrpoc.core.registry import Registry

from .program_input import Field


def make_instance_id(prefix: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in prefix.lower())
    return f"{safe}-{uuid4().hex[:12]}"


class DeviceError(Exception):
    """A device could not be reached or configured."""


# A dataclass with no fields, so every configuration type-checks as one.
@dataclass
class DeviceConfig:
    """Base for a device's settings: each field carries its rules in its metadata."""


def spec_field(default: Any, spec: Field, *, factory=None):
    """A dataclass field carrying the rules its values follow."""
    if factory is not None:
        return dc_field(default_factory=factory, metadata={"param": spec})
    return dc_field(default=default, metadata={"param": spec})


def config_fields(config: DeviceConfig | type[DeviceConfig]) -> list[tuple[str, Field]]:
    """``(name, rules)`` for every field of a configuration."""
    return [(f.name, f.metadata["param"]) for f in fields(config) if "param" in f.metadata]


def encode_config(config: DeviceConfig) -> dict[str, Any]:
    return {name: spec.encode(getattr(config, name)) for name, spec in config_fields(config)}


def decode_config(cls: type[DeviceConfig], raw: Mapping[str, Any]) -> DeviceConfig:
    """A configuration from a plain dict, checking every value. Missing keys
    take the field's default."""
    kwargs = {name: spec.decode(raw[name]) for name, spec in config_fields(cls) if name in raw}
    return cls(**kwargs)


class Connection(Enum):
    """How a device is reached: through a session the app keeps open from launch to exit, by a
    test the user can run, or not at all."""

    SESSION = "session"
    TESTABLE = "testable"
    NONE = "none"


class Device:
    """One addressable piece of the instrument."""

    display_name: str = "Device"
    registry_key: str = "device"

    connection: Connection = Connection.NONE

    # Set when this device has no connection of its own. Claims propagate up
    # this link, so claiming the galvo claims its DAQ.
    backed_by: type[Device] | None = None

    # The configuration describing this device's wiring and calibration.
    config_cls: type[DeviceConfig]

    def __init__(self, instance_id: str | None = None):
        self.instance_id = instance_id or make_instance_id(self.registry_key)
        self.config = self.config_cls()

    def connection_info(self) -> str:
        """Where the device is reached, like a port or serial number."""
        return ""

    def check_reachable(self) -> bool:
        """Raise or return False when a ``TESTABLE`` device is not there."""
        raise NotImplementedError

    @property
    def session_open(self) -> bool:
        return False

    def open_session(self) -> None:
        """Open the connection a ``SESSION`` device keeps."""
        raise NotImplementedError

    def close_session(self) -> None:
        """Close the connection a ``SESSION`` device keeps."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.instance_id}>"


device_registry: Registry[Device] = Registry("DeviceRegistry", Device)
