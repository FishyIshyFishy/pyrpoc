"""Two galvo pairs: four AO channels on the DAQ's card, for testing whether the
card can drive all four at once. Like the ``Galvo``, it is ``backed_by`` the DAQ."""

from __future__ import annotations

from dataclasses import dataclass

from pyrpoc.structs.plugins import params as P
from pyrpoc.structs.plugins.devices import Device, device_registry

from ..daq.device import DAQ


@dataclass
class DualGalvoConfig(P.Group):
    fast_ao: int = P.int_field(
        "Fast Axis AO",
        0,
        minimum=0,
        maximum=31,
        tooltip="Analog output channel for the scanning pair's fast (X) galvo",
    )
    slow_ao: int = P.int_field(
        "Slow Axis AO",
        1,
        minimum=0,
        maximum=31,
        tooltip="Analog output channel for the scanning pair's slow (Y) galvo",
    )
    second_fast_ao: int = P.int_field(
        "Second Fast Axis AO",
        2,
        minimum=0,
        maximum=31,
        tooltip="Analog output channel for the second pair's fast (X) galvo",
    )
    second_slow_ao: int = P.int_field(
        "Second Slow Axis AO",
        3,
        minimum=0,
        maximum=31,
        tooltip="Analog output channel for the second pair's slow (Y) galvo",
    )


@device_registry.register("dual_galvo")
class DualGalvo(Device):
    display_name = "Dual Galvos"
    backed_by = DAQ
    config_cls = DualGalvoConfig

    config: DualGalvoConfig

    @property
    def scan_channels(self) -> list[int]:
        return [self.config.fast_ao, self.config.slow_ao]

    @property
    def second_channels(self) -> list[int]:
        return [self.config.second_fast_ao, self.config.second_slow_ao]

    def summary(self) -> str:
        c = self.config
        return f"scan ao{c.fast_ao}/ao{c.slow_ao}, second ao{c.second_fast_ao}/ao{c.second_slow_ao}"
