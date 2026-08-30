"""The scanner: two AO channels on someone else's card.

No driver and no connection -- it is mirrors moved by voltages, so it is
``backed_by`` the DAQ and claiming it claims the DAQ. What it does have is a
wiring configuration you set up once and reuse, which is why it needs identity,
a panel and persistence.

Two fields, both read by ``raster_scan``. Section 4 also mentions per-axis
limits; nothing would clamp against them in v3.1, and an unused field is exactly
the accumulation section 3 warns about. They are a field plus a clamp when
something needs them.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.core import params as P

from ..base import Device
from ..daq.device import DAQ
from ..registry import device_registry


@dataclass
class GalvoConfig(P.Group):
    fast_ao: int = P.int_field(
        "Fast Axis AO", 0, minimum=0, maximum=31,
        tooltip="Analog output channel for the fast (X) galvo",
    )
    slow_ao: int = P.int_field(
        "Slow Axis AO", 1, minimum=0, maximum=31,
        tooltip="Analog output channel for the slow (Y) galvo",
    )


@device_registry.register("galvo")
class Galvo(Device):
    display_name = "Galvo"
    backed_by = DAQ
    config_cls = GalvoConfig

    config: GalvoConfig

    def summary(self) -> str:
        return f"fast ao{self.config.fast_ao}, slow ao{self.config.slow_ao}"
