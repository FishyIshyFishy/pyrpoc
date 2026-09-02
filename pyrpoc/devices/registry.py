from __future__ import annotations

from pyrpoc.core.registry import Registry

from .base import Device

device_registry: Registry[Device] = Registry("DeviceRegistry", Device)
