from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pyrpoc.acquisition.source import CommandSource, SourceDecorator
from pyrpoc.structs.commands import Command
from pyrpoc.structs.contexts import MaskContext
from .base import BaseOptoControl
from .registry import opto_control_registry


class MaskDecorator(SourceDecorator):
    """Attaches a drawn mask to each outgoing scan command.

    The handler turns the attached mask contexts into DAQ TTL output when it
    runs the command. Stacking multiple mask decorators appends each context.
    Commands that have no ``mask_contexts`` field (e.g. FLIM) are passed through
    untouched.
    """

    def __init__(self, inner: CommandSource, context: MaskContext):
        super().__init__(inner)
        self.context = context

    def transform(self, command: Command) -> Command:
        contexts = getattr(command, "mask_contexts", None)
        if contexts is not None:
            command.mask_contexts = list(contexts) + [self.context]  # type: ignore[attr-defined]
        return command


@opto_control_registry.register("mask")
class MaskOptoControl(BaseOptoControl):
    optocontrol_key = "mask"
    display_name = "Mask"

    def __init__(
        self,
        alias: str | None = None,
        user_label: str | None = None,
        enabled: bool = False,
        *,
        instance_id: str | None = None,
        connected: bool = False,
    ):
        super().__init__(
            alias=alias or self.optocontrol_key,
            user_label=user_label,
            enabled=enabled,
            instance_id=instance_id,
            connected=connected,
        )
        self.daq_port: int = 0
        self.daq_line: int = 0
        self.mask_path: str = ""
        self.mask_data: np.ndarray | None = None

    def get_context(self) -> MaskContext:
        self.context = MaskContext(
            optocontrol_key=self.optocontrol_key,
            alias=self.alias,
            mask=self.mask_data,
            daq_port=int(self.daq_port),
            daq_line=int(self.daq_line),
        )
        return self.context

    def build_decorator(self, inner: CommandSource) -> CommandSource:
        return MaskDecorator(inner, self.get_context())

    def export_persistence_state(self) -> dict[str, Any]:
        return {
            "daq_port": int(self.daq_port),
            "daq_line": int(self.daq_line),
            "mask_path": str(self.mask_path or "").strip(),
        }

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        self.daq_port = int(state.get("daq_port", self.daq_port))
        self.daq_line = int(state.get("daq_line", self.daq_line))
        self.mask_path = str(state.get("mask_path", self.mask_path) or "").strip()
        self.mask_data = None
        if self.mask_path:
            image = cv2.imread(str(Path(self.mask_path)), cv2.IMREAD_GRAYSCALE)
            if image is not None and image.ndim == 2:
                self.mask_data = image.astype(np.uint8, copy=True)
