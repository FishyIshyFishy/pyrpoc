from __future__ import annotations

from typing import Any

from pyrpoc.backend_utils.contracts import Action, Parameter
from .base_instrument import BaseInstrument
from .instrument_registry import instrument_registry


@instrument_registry.register("sim_galvo")
class SimGalvoInstrument(BaseInstrument):
    INSTRUMENT_KEY = "sim_galvo"
    DISPLAY_NAME = "Simulated Galvo"
    CONFIG_PARAMETERS = {
        "connection": [
            Parameter(
                label="Device Name",
                param_type=str,
                default="Dev1",
                tooltip="Simulated NI device identifier",
            ),
            Parameter(
                label="Sample Rate (Hz)",
                param_type=float,
                default=100000.0,
                minimum=1.0,
                tooltip="Simulated output sample rate",
            ),
        ]
    }
    ACTIONS = [
        Action(
            label="Home Galvos",
            method_name="home_galvos",
            parameters=[],
            tooltip="Set simulated galvo offsets to zero",
        ),
        Action(
            label="Set Offsets",
            method_name="set_offsets",
            parameters=[
                Parameter(
                    label="X Offset (V)",
                    param_type=float,
                    default=0.0,
                    minimum=-10.0,
                    maximum=10.0,
                ),
                Parameter(
                    label="Y Offset (V)",
                    param_type=float,
                    default=0.0,
                    minimum=-10.0,
                    maximum=10.0,
                ),
            ],
            tooltip="Set simulated galvo X/Y offsets",
        ),
    ]

    def __init__(self, alias: str | None = None):
        super().__init__(alias=alias)
        self.device_name = ""
        self.sample_rate = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0

    def connect(self, config: dict[str, Any]) -> None:
        self.device_name = str(config["Device Name"])
        self.sample_rate = float(config["Sample Rate (Hz)"])
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_status(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "connected": self._connected,
            "device_name": self.device_name,
            "sample_rate": self.sample_rate,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
        }

    def home_galvos(self, args: dict[str, Any]) -> None:
        del args
        self.x_offset = 0.0
        self.y_offset = 0.0

    def set_offsets(self, args: dict[str, Any]) -> None:
        self.x_offset = float(args["X Offset (V)"])
        self.y_offset = float(args["Y Offset (V)"])


Galvo = SimGalvoInstrument
