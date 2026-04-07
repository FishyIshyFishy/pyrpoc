from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import tifffile


def prepare_acquisition_storage(modality, *, frame_limit: int | None) -> None:
    modality._saved_frame_count = 0
    modality._run_frame_limit = frame_limit
    modality._run_id += 1
    modality._run_started_at = datetime.now(timezone.utc).isoformat()
    modality._save_channel_paths = {}
    modality._save_channel_labels = []
    modality._auxiliary_payload_buffers = {}
    modality._auxiliary_paths = {}
    if not modality._save_enabled:
        return
    if modality._save_root_path is None:
        raise RuntimeError("save_path is required when save_enabled is true")
    modality._save_root_path.parent.mkdir(parents=True, exist_ok=True)
    modality._save_json_path = modality._save_root_path.with_name(
        f"{modality._save_root_path.name}_meta.json"
    )
    write_metadata(modality, None)


def save_acquired_frame(modality, frame: np.ndarray, *, frame_index: int) -> None:
    if not modality._save_enabled:
        modality._pending_auxiliary = {}
        return
    if modality._save_root_path is None:
        raise RuntimeError("save_path is required when save_enabled is true")

    channel_data = modality._split_channels(frame)
    if not modality._save_channel_paths:
        labels = modality._resolve_channel_labels(len(channel_data))
        modality._save_channel_labels = labels
        modality._save_channel_paths = {
            label: modality._save_root_path.with_name(
                f"{modality._save_root_path.name}_{label}.tiff"
            )
            for label in labels
        }
        for path in modality._save_channel_paths.values():
            if path.exists():
                path.unlink()
    if len(channel_data) != len(modality._save_channel_paths):
        raise ValueError("frame channel count does not match configured save layout")

    for (label, path), channel_frame in zip(modality._save_channel_paths.items(), channel_data):
        del label
        with tifffile.TiffWriter(str(path), append=True) as writer:
            writer.write(channel_frame.astype(np.float32))

    append_auxiliary_payload(modality)
    modality._saved_frame_count = frame_index + 1
    write_metadata(modality, None)


def finalize_acquisition_storage(
    modality,
    *,
    frame_count: int,
    frame_limit: int | None,
    error: Exception | None,
) -> None:
    modality._saved_frame_count = frame_count
    modality._run_frame_limit = frame_limit
    flush_auxiliary_payloads(modality)
    write_metadata(modality, str(error) if error is not None else None)


def append_auxiliary_payload(modality) -> None:
    if not modality._save_enabled or modality._save_root_path is None:
        modality._pending_auxiliary = {}
        return
    if not modality._pending_auxiliary:
        return

    if not modality._auxiliary_paths:
        labels = list(modality._pending_auxiliary.keys())
        modality._auxiliary_paths = {
            label: modality._save_root_path.with_name(
                f"{modality._save_root_path.name}_{label}.npz"
            )
            for label in labels
        }
        for path in modality._auxiliary_paths.values():
            if path.exists():
                path.unlink()

    for label, payload in modality._pending_auxiliary.items():
        frames = modality._auxiliary_payload_buffers.setdefault(label, [])
        frames.append(np.asarray(payload, dtype=np.float32, copy=False))
    modality._pending_auxiliary = {}


def flush_auxiliary_payloads(modality) -> None:
    if not modality._save_enabled or not modality._auxiliary_paths:
        modality._pending_auxiliary = {}
        return
    for label, path in modality._auxiliary_paths.items():
        frames = modality._auxiliary_payload_buffers.get(label, [])
        if not frames:
            continue
        payload = np.stack(frames, axis=0)
        np.savez_compressed(
            str(path),
            frames=payload,
            parameters=np.asarray(modality._parameters_as_dict(), dtype=object),
            frame_indices=np.arange(payload.shape[0], dtype=np.int32),
        )
    modality._pending_auxiliary = {}


def write_metadata(modality, last_error: str | None) -> None:
    if not modality._save_enabled or modality._save_json_path is None:
        return
    payload = {
        "run_id": modality._run_id,
        "started": modality._run_started_at,
        "modality_key": modality.MODALITY_KEY,
        "save_root_path": str(modality._save_root_path),
        "save_json_path": str(modality._save_json_path),
        "tiff_paths": {label: str(path) for label, path in modality._save_channel_paths.items()},
        "auxiliary_paths": {label: str(path) for label, path in modality._auxiliary_paths.items()},
        "frames_saved": modality._saved_frame_count,
        "frame_limit": modality._run_frame_limit,
        "parameters": modality._parameters_as_dict(),
        "last_error": last_error,
    }
    modality._save_json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
