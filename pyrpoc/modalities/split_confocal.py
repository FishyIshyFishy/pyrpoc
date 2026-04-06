from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from pyrpoc.backend_utils.array_contracts import CONTRACT_CHW_FLOAT32
from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.backend_utils.opto_control_contexts import MaskContext
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl
from pyrpoc.optocontrols.mask import MaskOptoControl
from pyrpoc.backend_utils.parameter_utils import (
    CheckboxParameter,
    NumberParameter,
    PathParameter,
)

from .acquisition_functions.daq_helpers import (
    DaqUnavailableError,
    acquire_daq_split_confocal,
    extract_mask_contexts,
)
from .acquisition_functions.toy_data import generate_toy_split_confocal_frame
from .base_modality import BaseModality
from .mod_registry import modality_registry


@modality_registry.register("split_confocal")
class SplitConfocalModality(BaseModality):
    MODALITY_KEY = "split_confocal"
    DISPLAY_NAME = "Split Confocal"
    PARAMETERS = {
        "scan": [
            NumberParameter(
                label="X Pixels",
                default=512,
                minimum=8,
                tooltip="Number of pixels in X",
                number_type=int,
            ),
            NumberParameter(
                label="Y Pixels",
                default=512,
                minimum=8,
                tooltip="Number of pixels in Y",
                number_type=int,
            ),
            NumberParameter(
                label="Extra Steps Left",
                default=300,
                minimum=0,
                tooltip="Extra scan steps at left edge (stored only for now)",
                number_type=int,
            ),
            NumberParameter(
                label="Extra Steps Right",
                default=20,
                minimum=0,
                tooltip="Extra scan steps at right edge (stored only for now)",
                number_type=int,
            ),
            NumberParameter(
                label="Fast Axis Offset",
                default=0.0,
                tooltip="Fast-axis offset",
                number_type=float,
            ),
            NumberParameter(
                label="Fast Axis Amplitude",
                default=1.0,
                minimum=1e-6,
                tooltip="Fast-axis amplitude",
                number_type=float,
            ),
            NumberParameter(
                label="Slow Axis Offset",
                default=0.0,
                tooltip="Slow-axis offset",
                number_type=float,
            ),
            NumberParameter(
                label="Slow Axis Amplitude",
                default=1.0,
                minimum=1e-6,
                tooltip="Slow-axis amplitude",
                number_type=float,
            ),
            NumberParameter(
                label="Dwell Time (us)",
                default=2.0,
                minimum=0.1,
                tooltip="Pixel dwell time",
                number_type=float,
            ),
        ],
        "acquisition": [
            NumberParameter(
                label="t0 Samples",
                default=1,
                minimum=1,
                tooltip="Number of samples in the first subpixel window",
                number_type=int,
            ),
            NumberParameter(
                label="t1 Samples",
                default=0,
                minimum=0,
                tooltip="Number of samples to discard between t0 and t2",
                number_type=int,
            ),
            CheckboxParameter(
                label="save_enabled",
                display_label="save_enabled",
                default=False,
                required=False,
                tooltip="Enable saving frames and acquisition metadata",
            ),
            PathParameter(
                label="save_path",
                display_label="save_path",
                default="acquisition",
                required=False,
                tooltip="Base name/path for saved TIFF files (e.g. /dir/acquisition)",
            ),
            NumberParameter(
                label="num_frames",
                display_label="num_frames",
                default=1,
                required=False,
                minimum=1,
                tooltip="Number of frames to capture",
                number_type=int,
            ),
        ],
    }
    REQUIRED_INSTRUMENTS = [ConfocalDAQInstrument]
    OPTIONAL_INSTRUMENTS = []
    ALLOWED_OPTOCONTROLS = [MaskOptoControl]
    OUTPUT_DATA_CONTRACT = CONTRACT_CHW_FLOAT32
    ALLOWED_DISPLAYS = ["tiled_2d", "multichan_overlay"]

    def __init__(self):
        super().__init__()
        self._frame_idx = 0
        self._mask_contexts: list[MaskContext] = []
        self._daq_instrument: ConfocalDAQInstrument = None
        self._active_ai_channels: list[int] = []
        self._pending_auxiliary: dict[str, np.ndarray] = {}
        self._save_enabled = False
        self._save_root_path: Path | None = None
        self._save_json_path: Path | None = None
        self._save_channel_paths: dict[str, Path] = {}
        self._save_channel_labels: list[str] = []
        self._auxiliary_payload_buffers: dict[str, list[np.ndarray]] = {}
        self._auxiliary_paths: dict[str, Path] = {}
        self._saved_frame_count = 0
        self._run_id = 0
        self._run_started_at = ""
        self._run_frame_limit: int | None = 1

    def configure(
        self,
        params: dict[str, Any],
        instruments: dict[type[ConfocalDAQInstrument], ConfocalDAQInstrument],
        opto_controls: list[BaseOptoControl],
    ) -> None:
        self._params = dict(params)
        self._instruments = dict(instruments)
        instrument = self._instruments.get(ConfocalDAQInstrument)
        if instrument is None:
            raise RuntimeError("ConfocalDAQInstrument missing during configure")
        self._daq_instrument = instrument
        self._mask_contexts = extract_mask_contexts(opto_controls)
        self._active_ai_channels = self._get_active_ai_channels()
        self._frame_idx = 0
        self._pending_auxiliary = {}
        self._save_enabled = bool(self._params.get("save_enabled", False))
        self._save_root_path = self._coerce_save_root(self._params.get("save_path"))
        self._save_json_path = None
        self._save_channel_paths = {}
        self._save_channel_labels = []
        self._auxiliary_payload_buffers = {}
        self._auxiliary_paths = {}
        self._saved_frame_count = 0
        self._run_frame_limit = 1
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

        settings = self._get_scan_settings()
        split = self._get_split_settings()

        try:
            frame, raw = acquire_daq_split_confocal(
                daq_instrument=self._daq_instrument,
                active_ai_channels=self._active_ai_channels,
                mask_contexts=self._mask_contexts,
                **settings,
                **split,
            )
        except Exception as exc:
            print(f"acquisition unavailable ({exc}), falling back to toy data")
            if not isinstance(exc, DaqUnavailableError) and not isinstance(exc, RuntimeError):
                raise
            pixel_samples = max(1, int(settings["dwell_time_us"] * 1e-6 * float(self._daq_instrument.sample_rate_hz)))
            frame, raw = generate_toy_split_confocal_frame(
                x_pixels= settings["x_pixels"],
                y_pixels= settings["y_pixels"],
                active_channels=self._active_ai_channels,
                frame_index=self._frame_idx,
                mask_contexts=self._mask_contexts,
                fast_axis_offset=settings["fast_axis_offset"],
                fast_axis_amplitude=settings["fast_axis_amplitude"],
                slow_axis_offset=settings["slow_axis_offset"],
                slow_axis_amplitude=settings["slow_axis_amplitude"],
                t0_samples=split["t0_samples"],
                t1_samples=split["t1_samples"],
                pixel_samples=pixel_samples,
            )

        self._pending_auxiliary = {"raw_pixel_stream": raw}
        self._frame_idx += 1
        return frame.astype(np.float32, copy=False)

    def stop(self) -> None:
        self._running = False

    def prepare_acquisition_storage(self, *, frame_limit: int | None) -> None:
        self._saved_frame_count = 0
        self._run_frame_limit = frame_limit
        self._run_id += 1
        self._run_started_at = datetime.now(timezone.utc).isoformat()
        self._save_channel_paths = {}
        self._save_channel_labels = []
        self._auxiliary_payload_buffers = {}
        self._auxiliary_paths = {}
        if not self._save_enabled:
            return
        if self._save_root_path is None:
            raise RuntimeError("save_path is required when save_enabled is true")
        self._save_root_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_json_path = self._save_root_path.with_name(f"{self._save_root_path.name}_meta.json")
        self._write_metadata(None)

    def save_acquired_frame(self, frame: np.ndarray, *, frame_index: int) -> None:
        if not self._save_enabled:
            self._pending_auxiliary = {}
            return
        if self._save_root_path is None:
            raise RuntimeError("save_path is required when save_enabled is true")

        channel_data = self._split_channels(frame)
        if not self._save_channel_paths:
            labels = self._resolve_channel_labels(len(channel_data))
            self._save_channel_labels = labels
            self._save_channel_paths = {
                label: self._save_root_path.with_name(f"{self._save_root_path.name}_{label}.tiff")
                for label in labels
            }
            for path in self._save_channel_paths.values():
                if path.exists():
                    path.unlink()
        if len(channel_data) != len(self._save_channel_paths):
            raise ValueError("frame channel count does not match configured save layout")

        for (label, path), channel_frame in zip(self._save_channel_paths.items(), channel_data):
            del label
            with tifffile.TiffWriter(str(path), append=True) as writer:
                writer.write(channel_frame.astype(np.float32))

        self._append_auxiliary_payload()
        self._saved_frame_count = frame_index + 1
        self._write_metadata(None)

    def finalize_acquisition_storage(
        self,
        *,
        frame_count: int,
        frame_limit: int | None,
        error: Exception | None,
    ) -> None:
        self._saved_frame_count = frame_count
        self._run_frame_limit = frame_limit
        self._flush_auxiliary_payloads()
        self._write_metadata(str(error) if error is not None else None)

    def get_frame_limit(self) -> int | None:
        raw = self._params.get("num_frames", 1)
        limit = int(raw)
        if limit < 1:
            raise ValueError("num_frames must be >= 1")
        return limit

    def get_active_channel_labels(self) -> list[str]:
        labels: list[str] = []
        for channel in self._active_ai_channels:
            labels.append(f"ai{channel}_t0")
            labels.append(f"ai{channel}_t2")
        return labels

    def _get_active_ai_channels(self) -> list[int]:
        return [
            ai_channel
            for ai_channel, enabled in zip(
                self._daq_instrument.ai_channel_numbers if self._daq_instrument is not None else [],
                self._daq_instrument.active_ai_channels if self._daq_instrument is not None else [],
                strict=False,
            )
            if enabled
        ]

    def _get_scan_settings(self) -> dict[str, Any]:
        return {
            "x_pixels": int(self._params["X Pixels"]),
            "y_pixels": int(self._params["Y Pixels"]),
            "dwell_time_us": float(self._params["Dwell Time (us)"]),
            "extra_left": int(self._params["Extra Steps Left"]),
            "extra_right": int(self._params["Extra Steps Right"]),
            "fast_axis_offset": float(self._params["Fast Axis Offset"]),
            "fast_axis_amplitude": max(float(self._params["Fast Axis Amplitude"]), 1e-6),
            "slow_axis_offset": float(self._params["Slow Axis Offset"]),
            "slow_axis_amplitude": max(float(self._params["Slow Axis Amplitude"]), 1e-6),
        }

    def _get_split_settings(self) -> dict[str, Any]:
        t0 = int(self._params["t0 Samples"])
        t1 = int(self._params["t1 Samples"])
        if t0 < 1:
            raise ValueError("t0 Samples must be >= 1")
        if t1 < 0:
            raise ValueError("t1 Samples must be >= 0")
        return {
            "t0_samples": t0,
            "t1_samples": t1,
        }

    def _coerce_save_root(self, raw_save_path: Any) -> Path | None:
        if not self._save_enabled:
            return None
        if raw_save_path is None:
            raise ValueError("save_path is required when save_enabled is true")
        path = raw_save_path if isinstance(raw_save_path, Path) else Path(raw_save_path).expanduser()
        if path.suffix.lower() in {".tif", ".tiff"}:
            path = path.with_suffix("")
        return path

    def _resolve_channel_labels(self, channel_count: int) -> list[str]:
        active_labels = self.get_active_channel_labels()
        if active_labels and len(active_labels) == channel_count:
            return list(active_labels)
        return [f"channel_{index}" for index in range(channel_count)]

    def _split_channels(self, data: np.ndarray) -> list[np.ndarray]:
        if data.ndim == 2:
            return [data]
        if data.ndim == 3:
            return [data[index] for index in range(data.shape[0])]
        raise ValueError(f"unsupported frame dimensions {data.ndim}")

    def _append_auxiliary_payload(self) -> None:
        if not self._save_enabled or self._save_root_path is None:
            self._pending_auxiliary = {}
            return
        if not self._pending_auxiliary:
            return

        if not self._auxiliary_paths:
            labels = list(self._pending_auxiliary.keys())
            self._auxiliary_paths = {
                label: self._save_root_path.with_name(f"{self._save_root_path.name}_{label}.npz")
                for label in labels
            }
            for path in self._auxiliary_paths.values():
                if path.exists():
                    path.unlink()

        for label, payload in self._pending_auxiliary.items():
            frames = self._auxiliary_payload_buffers.setdefault(label, [])
            frames.append(np.asarray(payload, dtype=np.float32, copy=False))
        self._pending_auxiliary = {}

    def _flush_auxiliary_payloads(self) -> None:
        if not self._save_enabled or not self._auxiliary_paths:
            self._pending_auxiliary = {}
            return
        for label, path in self._auxiliary_paths.items():
            frames = self._auxiliary_payload_buffers.get(label, [])
            if not frames:
                continue
            payload = np.stack(frames, axis=0)
            np.savez_compressed(
                str(path),
                frames=payload,
                parameters=np.asarray(dict(self._params), dtype=object),
                frame_indices=np.arange(payload.shape[0], dtype=np.int32),
            )
        self._pending_auxiliary = {}

    def _write_metadata(self, last_error: str | None) -> None:
        if not self._save_enabled or self._save_json_path is None:
            return
        payload = {
            "run_id": self._run_id,
            "started": self._run_started_at,
            "modality_key": self.MODALITY_KEY,
            "save_root_path": str(self._save_root_path),
            "save_json_path": str(self._save_json_path),
            "tiff_paths": {label: str(path) for label, path in self._save_channel_paths.items()},
            "auxiliary_paths": {label: str(path) for label, path in self._auxiliary_paths.items()},
            "frames_saved": self._saved_frame_count,
            "frame_limit": self._run_frame_limit,
            "parameters": dict(self._params),
            "last_error": last_error,
        }
        self._save_json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


SplitConfocal = SplitConfocalModality
