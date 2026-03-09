from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pyrpoc.backend_utils.array_contracts import CONTRACT_CHW_FLOAT32
from pyrpoc.backend_utils.contracts import Parameter
from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.optocontrols.mask import MaskOptoControl
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from .base_modality import BaseModality
from .mod_registry import modality_registry


@modality_registry.register("confocal")
class ConfocalModality(BaseModality):
    MODALITY_KEY = "confocal"
    DISPLAY_NAME = "Confocal"
    PARAMETERS = {
        "scan": [
            Parameter(
                label="X Pixels",
                param_type=int,
                default=256,
                minimum=8,
                tooltip="Number of pixels in X",
            ),
            Parameter(
                label="Y Pixels",
                param_type=int,
                default=256,
                minimum=8,
                tooltip="Number of pixels in Y",
            ),
            Parameter(
                label="Extra Steps Left",
                param_type=int,
                default=0,
                minimum=0,
                tooltip="Extra scan steps at left edge (stored only for now)",
            ),
            Parameter(
                label="Extra Steps Right",
                param_type=int,
                default=0,
                minimum=0,
                tooltip="Extra scan steps at right edge (stored only for now)",
            ),
            Parameter(
                label="Fast Axis Offset",
                param_type=float,
                default=0.0,
                tooltip="Fast-axis offset",
            ),
            Parameter(
                label="Fast Axis Amplitude",
                param_type=float,
                default=1.0,
                minimum=1e-6,
                tooltip="Fast-axis amplitude",
            ),
            Parameter(
                label="Slow Axis Offset",
                param_type=float,
                default=0.0,
                tooltip="Slow-axis offset",
            ),
            Parameter(
                label="Slow Axis Amplitude",
                param_type=float,
                default=1.0,
                minimum=1e-6,
                tooltip="Slow-axis amplitude",
            ),
            Parameter(
                label="Dwell Time (us)",
                param_type=float,
                default=10.0,
                minimum=0.1,
                tooltip="Pixel dwell time",
            ),
        ],
    }
    REQUIRED_INSTRUMENTS = [ConfocalDAQInstrument]
    OPTIONAL_INSTRUMENTS = []
    ALLOWED_OPTOCONTROLS = [MaskOptoControl]
    OUTPUT_DATA_CONTRACT = CONTRACT_CHW_FLOAT32
    ALLOWED_DISPLAYS = ["tiled_2d"]

    def __init__(self):
        super().__init__()
        self._frame_idx = 0
        self._mask_contexts: list[dict[str, Any]] = []
        self._daq_instrument: ConfocalDAQInstrument | None = None

    def configure(
        self,
        params: dict[str, Any],
        instruments: dict[type[ConfocalDAQInstrument], ConfocalDAQInstrument],
        opto_controls: list[tuple[BaseOptoControl, tuple[Any, ...]]],
    ) -> None:
        self._params = dict(params)
        self._instruments = dict(instruments)
        instrument = self._instruments.get(ConfocalDAQInstrument)
        if instrument is None:
            raise RuntimeError("ConfocalDAQInstrument missing during configure")
        self._daq_instrument = instrument
        self._mask_contexts = self._extract_mask_contexts(opto_controls)
        self._configured = True

    def start(self) -> None:
        if not self._configured:
            raise RuntimeError("modality must be configured before start")
        if self._daq_instrument is None:
            raise RuntimeError("ConfocalDAQInstrument not configured")
        self._running = True

    def acquire_once(self) -> np.ndarray:
        if not self._running:
            raise RuntimeError("modality is not running")
        if self._daq_instrument is None:
            raise RuntimeError("ConfocalDAQInstrument unavailable")

        x_pixels = int(self._params["X Pixels"])
        y_pixels = int(self._params["Y Pixels"])
        active_channels = [
            ai
            for ai, enabled in zip(
                self._daq_instrument.ai_channel_numbers,
                self._daq_instrument.active_ai_channels,
                strict=False,
            )
            if enabled
        ]
        if not active_channels:
            raise RuntimeError("No active AI channels configured on ConfocalDAQInstrument")

        fast_offset = float(self._params["Fast Axis Offset"])
        fast_amp = max(float(self._params["Fast Axis Amplitude"]), 1e-6)
        slow_offset = float(self._params["Slow Axis Offset"])
        slow_amp = max(float(self._params["Slow Axis Amplitude"]), 1e-6)

        frame = np.zeros((len(active_channels), y_pixels, x_pixels), dtype=np.float32)
        for idx, ai_channel in enumerate(active_channels):
            frame[idx] = self._build_toy_channel(
                y_pixels=y_pixels,
                x_pixels=x_pixels,
                channel_index=idx,
                ai_channel=ai_channel,
                fast_offset=fast_offset,
                fast_amp=fast_amp,
                slow_offset=slow_offset,
                slow_amp=slow_amp,
            )

        self._apply_masks(frame)
        self._frame_idx += 1

        return frame.astype(np.float32, copy=False)

    def stop(self) -> None:
        self._running = False

    def _extract_mask_contexts(
        self,
        opto_controls: list[tuple[BaseOptoControl, tuple[Any, ...]]],
    ) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        for control, payload in opto_controls:
            if not isinstance(control, MaskOptoControl):
                continue
            if control.mask_data is None:
                raise RuntimeError(f"Enabled mask control '{control.alias}' has no mask data")
            mask = np.asarray(control.mask_data, dtype=np.uint8)
            if mask.ndim != 2:
                raise RuntimeError(f"Mask control '{control.alias}' must provide a 2D mask")

            daq_port = int(control.daq_port)
            daq_line = int(control.daq_line)
            if len(payload) >= 2 and isinstance(payload[1], dict):
                daq_port = int(payload[1].get("daq_port", daq_port))
                daq_line = int(payload[1].get("daq_line", daq_line))

            if daq_port < 0 or daq_line < 0:
                raise RuntimeError(f"Mask control '{control.alias}' has invalid DAQ port/line values")

            contexts.append(
                {
                    "alias": control.alias,
                    "daq_port": daq_port,
                    "daq_line": daq_line,
                    "mask": mask,
                }
            )
        return contexts

    def _build_toy_channel(
        self,
        y_pixels: int,
        x_pixels: int,
        channel_index: int,
        ai_channel: int,
        fast_offset: float,
        fast_amp: float,
        slow_offset: float,
        slow_amp: float,
    ) -> np.ndarray:
        seed = (self._frame_idx + 1) * 1009 + (channel_index + 1) * 101 + ai_channel * 17
        rng = np.random.default_rng(seed)
        channel = np.zeros((y_pixels, x_pixels), dtype=np.float32)

        # Build a low-frequency background that varies per channel and frame.
        x = np.linspace(-1.0, 1.0, x_pixels, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, y_pixels, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        x_term = (xx - fast_offset) / fast_amp
        y_term = (yy - slow_offset) / slow_amp
        channel += 0.15 * np.sin((x_term + 0.07 * self._frame_idx) * (5.0 + channel_index))
        channel += 0.12 * np.cos((y_term - 0.05 * self._frame_idx) * (4.0 + 0.5 * channel_index))

        n_shapes = int(rng.integers(10, 18))
        for _ in range(n_shapes):
            intensity = float(rng.uniform(0.2, 1.0))
            if rng.random() < 0.5:
                cx = int(rng.integers(0, x_pixels))
                cy = int(rng.integers(0, y_pixels))
                radius = int(rng.integers(max(3, min(x_pixels, y_pixels) // 30), max(6, min(x_pixels, y_pixels) // 8)))
                cv2.circle(channel, (cx, cy), radius, intensity, thickness=-1, lineType=cv2.LINE_AA)
            else:
                cx = int(rng.integers(0, x_pixels))
                cy = int(rng.integers(0, y_pixels))
                a = int(rng.integers(max(4, x_pixels // 40), max(8, x_pixels // 10)))
                b = int(rng.integers(max(4, y_pixels // 40), max(8, y_pixels // 10)))
                angle = float(rng.uniform(0.0, 180.0))
                cv2.ellipse(
                    channel,
                    (cx, cy),
                    (a, b),
                    angle,
                    0.0,
                    360.0,
                    intensity,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        channel += rng.normal(0.0, 0.03, size=(y_pixels, x_pixels)).astype(np.float32)
        channel -= float(np.min(channel))
        peak = float(np.max(channel))
        if peak > 0:
            channel /= peak
        return channel.astype(np.float32, copy=False)

    def _apply_masks(self, frame: np.ndarray) -> None:
        if frame.ndim != 3:
            raise ValueError("confocal frame must be [C,H,W]")
        _, h, w = frame.shape
        for context in self._mask_contexts:
            mask = np.asarray(context["mask"], dtype=np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            active = mask > 0
            if not np.any(active):
                continue
            for idx in range(frame.shape[0]):
                boost = float(np.max(frame[idx]))
                frame[idx, active] += boost

Confocal = ConfocalModality
