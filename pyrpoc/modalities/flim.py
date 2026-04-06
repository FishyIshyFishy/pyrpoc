from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np
import tifffile

from pyrpoc.backend_utils.array_contracts import CONTRACT_CHW_FLOAT32
from pyrpoc.instruments.confocal_daq import ConfocalDAQInstrument
from pyrpoc.instruments.time_tagger import TimeTaggerInstrument
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
    acquire_daq_flim,
    extract_mask_contexts,
)
from .acquisition_functions.flim_helpers import poll_one_flim_frame
from .acquisition_functions.toy_data import generate_toy_confocal_frame
from .base_modality import BaseModality
from .mod_registry import modality_registry


@modality_registry.register("flim")
class FlimModality(BaseModality):
    MODALITY_KEY = "flim"
    DISPLAY_NAME = "FLIM"
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
                tooltip="Extra scan steps at left edge",
                number_type=int,
            ),
            NumberParameter(
                label="Extra Steps Right",
                default=20,
                minimum=0,
                tooltip="Extra scan steps at right edge",
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
        "timetagger": [
            NumberParameter(
                label="Laser Channel",
                default=1,
                minimum=1,
                tooltip="TimeTagger input channel for laser sync",
                number_type=int,
            ),
            NumberParameter(
                label="Detector Channel",
                default=2,
                minimum=1,
                tooltip="TimeTagger input channel for SPAD detector",
                number_type=int,
            ),
            NumberParameter(
                label="Pixel Clock Channel",
                default=3,
                minimum=1,
                tooltip="TimeTagger input channel for pixel clock",
                number_type=int,
            ),
            NumberParameter(
                label="Pixel Clock DO Line",
                default=0,
                minimum=0,
                tooltip="DAQ port0 line number to output the pixel clock TTL",
                number_type=int,
            ),
            NumberParameter(
                label="Laser Frequency MHz",
                default=80.0,
                minimum=0.001,
                tooltip="Laser repetition rate in MHz (used to fold delays)",
                number_type=float,
            ),
            NumberParameter(
                label="Laser Trigger V",
                default=0.05,
                tooltip="Trigger threshold for laser sync channel (V)",
                number_type=float,
            ),
            NumberParameter(
                label="Detector Trigger V",
                default=0.2,
                tooltip="Trigger threshold for SPAD detector channel (V)",
                number_type=float,
            ),
            NumberParameter(
                label="Pixel Trigger V",
                default=0.2,
                tooltip="Trigger threshold for pixel clock channel (V)",
                number_type=float,
            ),
            NumberParameter(
                label="Laser Event Divider",
                default=1,
                minimum=1,
                tooltip="Only keep 1 in N laser sync events (reduces data rate)",
                number_type=int,
            ),
        ],
        "acquisition": [
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
                tooltip="Base name/path for saved files",
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
    REQUIRED_INSTRUMENTS = [ConfocalDAQInstrument, TimeTaggerInstrument]
    OPTIONAL_INSTRUMENTS = []
    ALLOWED_OPTOCONTROLS = [MaskOptoControl]
    OUTPUT_DATA_CONTRACT = CONTRACT_CHW_FLOAT32
    ALLOWED_DISPLAYS = ["tiled_2d", "multichan_overlay"]

    def __init__(self):
        super().__init__()
        self._frame_idx = 0
        self._mask_contexts: list[MaskContext] = []
        self._daq_instrument: ConfocalDAQInstrument | None = None
        self._tagger_instrument: TimeTaggerInstrument | None = None
        self._stream = None
        self._active_ai_channels: list[int] = []
        self._pending_flim_frame: np.ndarray | None = None
        self._raw_frames: list[np.ndarray] = []
        self._save_enabled = False
        self._save_root_path: Path | None = None
        self._save_json_path: Path | None = None
        self._save_channel_paths: dict[str, Path] = {}
        self._save_channel_labels: list[str] = []
        self._saved_frame_count = 0
        self._run_id = 0
        self._run_started_at = ""
        self._run_frame_limit: int | None = 1

    def configure(
        self,
        params: dict[str, Any],
        instruments: dict[type, Any],
        opto_controls: list[BaseOptoControl],
    ) -> None:
        self._params = dict(params)
        self._instruments = dict(instruments)
        self._daq_instrument = self._instruments.get(ConfocalDAQInstrument)
        self._tagger_instrument = self._instruments.get(TimeTaggerInstrument)
        if self._daq_instrument is None:
            raise RuntimeError("ConfocalDAQInstrument missing during configure")
        if self._tagger_instrument is None:
            raise RuntimeError("TimeTaggerInstrument missing during configure")
        self._mask_contexts = extract_mask_contexts(opto_controls)
        self._active_ai_channels = self._get_active_ai_channels()
        self._frame_idx = 0
        self._save_enabled = bool(self._params.get("save_enabled", False))
        self._save_root_path = self._coerce_save_root(self._params.get("save_path"))
        self._save_json_path = None
        self._save_channel_paths = {}
        self._save_channel_labels = []
        self._saved_frame_count = 0
        self._run_frame_limit = 1
        self._configured = True

    def start(self) -> None:
        if not self._configured:
            raise RuntimeError("modality must be configured before start")
        self._stream = None
        laser_ch = int(self._params["Laser Channel"])
        detector_ch = int(self._params["Detector Channel"])
        pixel_ch = int(self._params["Pixel Clock Channel"])
        self._tagger_instrument.create_tagger()
        self._tagger_instrument.configure_for_flim(
            laser_ch=laser_ch,
            detector_ch=detector_ch,
            pixel_ch=pixel_ch,
            laser_trigger_v=float(self._params["Laser Trigger V"]),
            detector_trigger_v=float(self._params["Detector Trigger V"]),
            pixel_trigger_v=float(self._params["Pixel Trigger V"]),
            laser_event_divider=int(self._params["Laser Event Divider"]),
        )
        self._stream = self._tagger_instrument.create_flim_stream(
            laser_ch=laser_ch,
            detector_ch=detector_ch,
            pixel_ch=pixel_ch,
        )
        self._stream.start()
        self._running = True

    def acquire_once(self) -> np.ndarray:
        if not self._running:
            raise RuntimeError("modality is not running")

        scan = self._get_scan_settings()
        active_ai = self._get_active_ai_channels()
        pixel_dwell_ps = int(round(scan["dwell_time_us"] * 1e6))
        scan_duration_s = (
            (scan["x_pixels"] + scan["extra_left"] + scan["extra_right"])
            * scan["y_pixels"]
            * scan["dwell_time_us"]
            * 1e-6
        )
        self._pending_flim_frame = None

        laser_ch = int(self._params["Laser Channel"])
        detector_ch = int(self._params["Detector Channel"])
        pixel_ch = int(self._params["Pixel Clock Channel"])

        poll_result: dict[str, Any] = {}

        def _poll() -> None:
            try:
                poll_result["frame"] = poll_one_flim_frame(
                    stream=self._stream,
                    x_pixels=scan["x_pixels"],
                    y_pixels=scan["y_pixels"],
                    extra_left=scan["extra_left"],
                    extra_right=scan["extra_right"],
                    pixel_dwell_ps=pixel_dwell_ps,
                    laser_ch=laser_ch,
                    detector_ch=detector_ch,
                    pixel_ch=pixel_ch,
                )
            except Exception as exc:
                poll_result["error"] = exc

        poll_thread = threading.Thread(target=_poll, daemon=True)
        poll_thread.start()

        try:
            daq_frame = acquire_daq_flim(
                daq_instrument=self._daq_instrument,
                x_pixels=scan["x_pixels"],
                y_pixels=scan["y_pixels"],
                extra_left=scan["extra_left"],
                extra_right=scan["extra_right"],
                dwell_time_us=scan["dwell_time_us"],
                fast_axis_offset=scan["fast_axis_offset"],
                fast_axis_amplitude=scan["fast_axis_amplitude"],
                slow_axis_offset=scan["slow_axis_offset"],
                slow_axis_amplitude=scan["slow_axis_amplitude"],
                active_ai_channels=active_ai,
                mask_contexts=self._mask_contexts,
                pixel_clock_do_line=int(self._params["Pixel Clock DO Line"]),
            )
        except DaqUnavailableError:
            self._emit_warning("DAQ unavailable — displaying simulated data")
            daq_frame = generate_toy_confocal_frame(
                x_pixels=scan["x_pixels"],
                y_pixels=scan["y_pixels"],
                active_channels=active_ai if active_ai else [0],
                frame_index=self._frame_idx,
                mask_contexts=self._mask_contexts,
                fast_axis_offset=scan["fast_axis_offset"],
                fast_axis_amplitude=scan["fast_axis_amplitude"],
                slow_axis_offset=scan["slow_axis_offset"],
                slow_axis_amplitude=scan["slow_axis_amplitude"],
            )

        self._frame_idx += 1

        poll_thread.join(timeout=scan_duration_s * 2 + 5)
        if "error" in poll_result:
            raise RuntimeError(f"TimeTagger poll failed: {poll_result['error']}") from poll_result["error"]

        flim_frame = poll_result.get("frame")
        self._pending_flim_frame = flim_frame
        if flim_frame is not None:
            intensity = np.vectorize(len)(flim_frame).astype(np.float32)
            return intensity[np.newaxis].astype(np.float32)

        raise RuntimeError("TimeTagger poll did not return a frame within the expected window")

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream = None
        if self._tagger_instrument is not None:
            self._tagger_instrument.free_tagger()
        self._timetagger_available = False

    def prepare_acquisition_storage(self, *, frame_limit: int | None) -> None:
        self._saved_frame_count = 0
        self._run_frame_limit = frame_limit
        self._run_id += 1
        self._run_started_at = datetime.now(timezone.utc).isoformat()
        self._save_channel_paths = {}
        self._save_channel_labels = []
        self._raw_frames = []
        if not self._save_enabled:
            return
        if self._save_root_path is None:
            raise RuntimeError("save_path is required when save_enabled is true")
        self._save_root_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_json_path = self._save_root_path.with_name(f"{self._save_root_path.name}_meta.json")
        self._write_metadata(None)

    def save_acquired_frame(self, frame: np.ndarray, *, frame_index: int) -> None:
        if self._pending_flim_frame is not None:
            self._raw_frames.append(self._pending_flim_frame)

        if not self._save_enabled:
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

        for (label, path), channel_frame in zip(self._save_channel_paths.items(), channel_data):
            del label
            with tifffile.TiffWriter(str(path), append=True) as writer:
                writer.write(channel_frame.astype(np.float32))
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
        self._write_metadata(str(error) if error is not None else None)
        if self._save_enabled and self._save_root_path is not None and self._raw_frames:
            npz_path = self._save_root_path.with_name(f"{self._save_root_path.name}_raw.npz")
            np.savez_compressed(
                npz_path,
                frames=np.asarray(self._raw_frames, dtype=object),
                acquisition_parameters=np.asarray(dict(self._params), dtype=object),
            )

    def get_frame_limit(self) -> int | None:
        raw = self._params.get("num_frames", 1)
        limit = int(raw)
        if limit < 1:
            raise ValueError("num_frames must be >= 1")
        return limit

    def get_active_channel_labels(self) -> list[str]:
        return ["intensity"]

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
        return [f"channel_{i}" for i in range(channel_count)]

    def _split_channels(self, data: np.ndarray) -> list[np.ndarray]:
        if data.ndim == 2:
            return [data]
        if data.ndim == 3:
            return [data[i] for i in range(data.shape[0])]
        raise ValueError(f"unsupported frame dimensions {data.ndim}")

    def _write_metadata(self, last_error: str | None) -> None:
        if not self._save_enabled or self._save_json_path is None:
            return
        payload = {
            "run_id": self._run_id,
            "started": self._run_started_at,
            "modality_key": self.MODALITY_KEY,
            "save_root_path": str(self._save_root_path),
            "tiff_paths": {label: str(path) for label, path in self._save_channel_paths.items()},
            "frames_saved": self._saved_frame_count,
            "frame_limit": self._run_frame_limit,
            "parameters": dict(self._params),
            "last_error": last_error,
        }
        self._save_json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _get_active_ai_channels(self) -> list[int]:
        if self._daq_instrument is None:
            return []
        return [
            ch
            for ch, enabled in zip(
                self._daq_instrument.ai_channel_numbers,
                self._daq_instrument.active_ai_channels,
                strict=False,
            )
            if enabled
        ]

    def _get_scan_settings(self) -> dict[str, Any]:
        return {
            "x_pixels": int(self._params["X Pixels"]),
            "y_pixels": int(self._params["Y Pixels"]),
            "extra_left": int(self._params["Extra Steps Left"]),
            "extra_right": int(self._params["Extra Steps Right"]),
            "dwell_time_us": float(self._params["Dwell Time (us)"]),
            "fast_axis_offset": float(self._params["Fast Axis Offset"]),
            "fast_axis_amplitude": max(float(self._params["Fast Axis Amplitude"]), 1e-6),
            "slow_axis_offset": float(self._params["Slow Axis Offset"]),
            "slow_axis_amplitude": max(float(self._params["Slow Axis Amplitude"]), 1e-6),
        }


Flim = FlimModality
