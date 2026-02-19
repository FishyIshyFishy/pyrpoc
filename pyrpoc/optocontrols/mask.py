from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pyrpoc.backend_utils.contracts import Action, Parameter
from .base_optocontrol import BaseOptoControl
from .opto_control_registry import opto_control_registry


@opto_control_registry.register("mask")
class MaskOptoControl(BaseOptoControl):
    OPTOCONTROL_KEY = "mask"
    DISPLAY_NAME = "Mask Opto-Control"
    EDITOR_KEY = "mask"
    EDITOR_ANCHOR_PARAM = "Mask Path"
    EDITOR_APPLY_METHOD = "set_mask_data"
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
            parameters=[
                Parameter(
                    label="Mask Path",
                    param_type=Path,
                    default=None,
                    required=False,
                    tooltip="Optional image path; if omitted, uses configured Mask Path",
                )
            ],
            tooltip="Load a mask image from file",
        ),
        Action(
            label="Create Mask",
            method_name="create_mask",
            parameters=[],
            tooltip="Mark mask as created (editor integration handled by UI)",
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
        self.mask_data: np.ndarray | None = None
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
            self._load_mask_from_path(self.mask_path)
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
            "mask_shape": None if self.mask_data is None else tuple(self.mask_data.shape),
            "enabled": self.enabled,
            "last_action": self.last_action,
        }

    def load_mask_image(self, args: dict[str, Any]) -> None:
        raw_mask_path = args.get("Mask Path")
        if raw_mask_path is None and self.mask_path is None:
            raise RuntimeError("no mask path provided")

        path = Path(raw_mask_path) if raw_mask_path is not None else self.mask_path
        if path is None:
            raise RuntimeError("no mask path provided")

        self._load_mask_from_path(path)
        self.last_action = "load_mask_image"

    def create_mask(self, args: dict[str, Any]) -> None:
        del args
        self.has_mask = True
        self.last_action = "create_mask"

    def clear_mask(self, args: dict[str, Any]) -> None:
        del args
        self.mask_data = None
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

    def on_enabled_changed(self, enabled: bool) -> None:
        if enabled and not self.has_mask:
            raise RuntimeError("cannot enable mask before a mask is loaded or created")
        self.enabled = bool(enabled)
        self.last_action = "enable_mask" if self.enabled else "disable_mask"

    def set_mask_data(self, mask_data: np.ndarray, source_path: str | Path | None = None) -> None:
        if not isinstance(mask_data, np.ndarray):
            raise TypeError("mask_data must be a numpy array")
        if mask_data.ndim != 2:
            raise ValueError("mask_data must be a 2D array")

        if mask_data.dtype != np.uint8:
            mask_data = np.clip(mask_data, 0, 255).astype(np.uint8)

        self.mask_data = mask_data
        self.has_mask = True
        if source_path is not None:
            self.mask_path = Path(source_path)
        self.last_action = "create_mask"

    def _load_mask_from_path(self, path: Path) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"failed to load mask image from '{path}'")
        self.mask_data = image
        self.mask_path = path
        self.has_mask = True


Mask = MaskOptoControl
