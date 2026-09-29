from __future__ import annotations

from pyrpoc.structs.registries import Registry

from .base import Device

device_registry: Registry[Device] = Registry("DeviceRegistry", Device)
