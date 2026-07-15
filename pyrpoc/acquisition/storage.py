from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from pyrpoc.structs.acquired_data import AcquiredData, DataKind


def split_channels(data: np.ndarray) -> list[np.ndarray]:
    if data.ndim == 2:
        return [data]
    if data.ndim == 3:
        return [data[index] for index in range(data.shape[0])]
    raise ValueError(f"unsupported frame dimensions {data.ndim}")


def resolve_channel_labels(labels: list[str], channel_count: int) -> list[str]:
    if labels and len(labels) == channel_count:
        return list(labels)
    return [f"channel_{index}" for index in range(channel_count)]


class FrameStorage:
    """Writes acquired results to disk, independent of any modality object.

    Intensity frames become per-channel appended TIFF stacks; ``FLIM_RAW_FRAME``
    results and any ``metadata['auxiliary']`` arrays are buffered and flushed to
    ``.npz`` at finalize. A ``_meta.json`` sidecar tracks the run.
    """

    def __init__(self, *, save_enabled: bool, save_path: str, preset_key: str, parameters: dict[str, Any]):
        self.save_enabled = bool(save_enabled)
        self.preset_key = preset_key
        self.parameters = dict(parameters)
        self.root: Path | None = None
        if self.save_enabled:
            path = Path(save_path).expanduser()
            if path.suffix.lower() in {".tif", ".tiff"}:
                path = path.with_suffix("")
            self.root = path
        self.json_path: Path | None = None
        self.channel_paths: dict[str, Path] = {}
        self.aux_buffers: dict[str, list[np.ndarray]] = {}
        self.flim_raw_buffer: list[np.ndarray] = []
        self.saved_frame_count = 0
        self.frame_limit: int | None = None
        self.run_started_at = ""
        self.last_error: str | None = None

    def prepare(self, *, frame_limit: int | None) -> None:
        self.frame_limit = frame_limit
        self.saved_frame_count = 0
        self.channel_paths = {}
        self.aux_buffers = {}
        self.flim_raw_buffer = []
        self.run_started_at = datetime.now(timezone.utc).isoformat()
        if not self.save_enabled or self.root is None:
            return
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.json_path = self.root.with_name(f"{self.root.name}_meta.json")
        self.write_metadata(None)

    def save(self, results: list[AcquiredData]) -> None:
        if not self.save_enabled or self.root is None:
            return
        for result in results:
            if result.kind == DataKind.FLIM_RAW_FRAME:
                self.flim_raw_buffer.append(np.asarray(result.data, dtype=np.float32))
        intensity = [r for r in results if r.kind == DataKind.INTENSITY_FRAME]
        for result in intensity:
            self.save_intensity_frame(result)
            self.buffer_auxiliary(result)
            self.saved_frame_count += 1
        self.write_metadata(None)

    def save_intensity_frame(self, result: AcquiredData) -> None:
        assert self.root is not None
        channel_data = split_channels(np.asarray(result.data, dtype=np.float32))
        if not self.channel_paths:
            labels = resolve_channel_labels(result.channel_labels, len(channel_data))
            self.channel_paths = {
                label: self.root.with_name(f"{self.root.name}_{label}.tiff") for label in labels
            }
            for path in self.channel_paths.values():
                if path.exists():
                    path.unlink()
        if len(channel_data) != len(self.channel_paths):
            raise ValueError("frame channel count does not match configured save layout")
        for path, channel_frame in zip(self.channel_paths.values(), channel_data):
            with tifffile.TiffWriter(str(path), append=True) as writer:
                writer.write(channel_frame.astype(np.float32))

    def buffer_auxiliary(self, result: AcquiredData) -> None:
        auxiliary = result.metadata.get("auxiliary")
        if not isinstance(auxiliary, dict):
            return
        for label, payload in auxiliary.items():
            self.aux_buffers.setdefault(str(label), []).append(np.asarray(payload, dtype=np.float32))

    def finalize(self, error: Exception | None) -> None:
        self.last_error = str(error) if error is not None else None
        if not self.save_enabled or self.root is None:
            return
        for label, frames in self.aux_buffers.items():
            if frames:
                np.savez_compressed(
                    str(self.root.with_name(f"{self.root.name}_{label}.npz")),
                    frames=np.stack(frames, axis=0),
                    parameters=np.asarray(self.parameters, dtype=object),
                )
        if self.flim_raw_buffer:
            np.savez_compressed(
                str(self.root.with_name(f"{self.root.name}_flim_raw.npz")),
                frames=np.stack(self.flim_raw_buffer, axis=0),
                parameters=np.asarray(self.parameters, dtype=object),
            )
        self.write_metadata(self.last_error)

    def write_metadata(self, last_error: str | None) -> None:
        if not self.save_enabled or self.json_path is None or self.root is None:
            return
        payload = {
            "started": self.run_started_at,
            "preset_key": self.preset_key,
            "save_root_path": str(self.root),
            "tiff_paths": {label: str(path) for label, path in self.channel_paths.items()},
            "frames_saved": self.saved_frame_count,
            "frame_limit": self.frame_limit,
            "parameters": self.parameters,
            "last_error": last_error,
        }
        self.json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
