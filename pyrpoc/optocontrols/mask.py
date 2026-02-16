from __future__ import annotations

from pathlib import Path
from typing import Any

from pyrpoc.backend_utils.contracts import Action, Parameter
from .base_optocontrol import BaseOptoControl
from .opto_control_registry import opto_control_registry


@opto_control_registry.register("mask")
class MaskOptoControl(BaseOptoControl):
    OPTOCONTROL_KEY = "mask"
    DISPLAY_NAME = "Mask Opto-Control"
    CONFIG_PARAMETERS = {
        "connection": [
            Parameter(
                label="DAQ DO Channel",
                param_type=str,
                default="Dev1/port0/line0",
                tooltip="DAQ digital output line used to gate opto-control",
            ),
            Parameter(
                label="Mask Path",
                param_type=Path,
                default=None,
                required=False,
                tooltip="Optional mask file path for initial load",
            ),
        ]
    }
    ACTIONS = [
        Action(
            label="Load Mask Image",
            method_name="load_mask_image",
            parameters=[],
            tooltip="Placeholder action to load a mask image",
        ),
        Action(
            label="Create Mask",
            method_name="create_mask",
            parameters=[],
            tooltip="Placeholder action to create a new mask",
        ),
        Action(
            label="Clear Mask",
            method_name="clear_mask",
            parameters=[],
            tooltip="Clear current mask state",
        ),
        Action(
            label="Enable Mask",
            method_name="enable_mask",
            parameters=[],
            tooltip="Enable mask output",
        ),
        Action(
            label="Disable Mask",
            method_name="disable_mask",
            parameters=[],
            tooltip="Disable mask output",
        ),
    ]

    def __init__(self, alias: str | None = None):
        super().__init__(alias=alias)
        self.daq_do_channel = ""
        self.mask_path: Path | None = None
        self.has_mask = False
        self.enabled = False
        self.last_action = "idle"

    def connect(self, config: dict[str, Any]) -> None:
        self.daq_do_channel = str(config["DAQ DO Channel"])
        raw_mask_path = config.get("Mask Path")
        if raw_mask_path is None or str(raw_mask_path).strip() == "":
            self.mask_path = None
        else:
            self.mask_path = Path(raw_mask_path)
            self.has_mask = True
        self._connected = True
        self.last_action = "configured"

    def disconnect(self) -> None:
        self.enabled = False
        self._connected = False
        self.last_action = "disconnected"

    def get_status(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "connected": self._connected,
            "daq_do_channel": self.daq_do_channel,
            "mask_path": None if self.mask_path is None else str(self.mask_path),
            "has_mask": self.has_mask,
            "enabled": self.enabled,
            "last_action": self.last_action,
        }

    def load_mask_image(self, args: dict[str, Any]) -> None:
        del args
        self.has_mask = True
        self.last_action = "load_mask_image (placeholder)"

    def create_mask(self, args: dict[str, Any]) -> None:
        del args
        self.has_mask = True
        self.last_action = "create_mask (placeholder)"

    def clear_mask(self, args: dict[str, Any]) -> None:
        del args
        self.has_mask = False
        self.enabled = False
        self.last_action = "clear_mask"

    def enable_mask(self, args: dict[str, Any]) -> None:
        del args
        if not self.has_mask:
            raise RuntimeError("cannot enable mask before a mask is loaded or created")
        self.enabled = True
        self.last_action = "enable_mask"

    def disable_mask(self, args: dict[str, Any]) -> None:
        del args
        self.enabled = False
        self.last_action = "disable_mask"


Mask = MaskOptoControl
