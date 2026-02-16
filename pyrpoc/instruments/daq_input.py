from __future__ import annotations

from typing import Any

from pyrpoc.backend_utils.contracts import Parameter
from .base_instrument import BaseInstrument
from .instrument_registry import instrument_registry


@instrument_registry.register("sim_daq_input")
class SimDAQInput(BaseInstrument):
    INSTRUMENT_KEY = "sim_daq_input"
    DISPLAY_NAME = "Simulated DAQ Input"
    CONFIG_PARAMETERS = {
        "connection": [
            Parameter(
                label="Device Name",
                param_type=str,
                default="Dev1",
            )
        ]
    }
    ACTIONS = []

    def __init__(self, alias: str | None = None):
        super().__init__(alias=alias)
        self.device_name = ""

    def connect(self, config: dict[str, Any]) -> None:
        self.device_name = str(config["Device Name"])
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_status(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "connected": self._connected,
            "device_name": self.device_name,
        }


DAQInput = SimDAQInput
