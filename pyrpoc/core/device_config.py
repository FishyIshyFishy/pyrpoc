"""``DeviceConfig``: a device's settings, as a dataclass whose fields carry
the rules their values follow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from dataclasses import field as dc_field
from typing import Any

from .program_inputs import ChannelsField, Field, FloatField, IntField, PathField, TextField


# A dataclass with no fields, so every configuration type-checks as one.
@dataclass
class DeviceConfig:
    """Base for a device's settings: each field carries its rules in its metadata."""


def spec_field(default: Any, spec: Field, *, factory=None):
    """A dataclass field carrying the rules its values follow."""
    if factory is not None:
        return dc_field(default_factory=factory, metadata={"param": spec})
    return dc_field(default=default, metadata={"param": spec})


def int_field(label, default, *, minimum=None, maximum=None, step=1, tooltip=""):
    return spec_field(default, IntField(label, tooltip, minimum, maximum, step))


def float_field(label, default, *, minimum=None, maximum=None, step=0.1, decimals=6, tooltip=""):
    return spec_field(default, FloatField(label, tooltip, minimum, maximum, step, decimals))


def text_field(label, default="", *, tooltip=""):
    return spec_field(default, TextField(label, tooltip))


def path_field(label, default="", *, dialog_filter="All Files (*)", tooltip=""):
    return spec_field(default, PathField(label, tooltip, dialog_filter))


def channels_field(label, *, num_channels=9, default=None, tooltip=""):
    resolved = tuple(range(num_channels)) if default is None else tuple(default)
    return spec_field(None, ChannelsField(label, tooltip, num_channels), factory=lambda: resolved)


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
