"""``DeviceConfig``: a device's settings, as a dataclass whose fields carry
the rules their values follow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from dataclasses import field as dc_field
from typing import Any

from .program_input import Field


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
