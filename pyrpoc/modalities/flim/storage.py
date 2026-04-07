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
    modality._raw_frames = []
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
    if modality._pending_flim_frame is not None:
        modality._raw_frames.append(modality._pending_flim_frame)

    if not modality._save_enabled:
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

    for (label, path), channel_frame in zip(modality._save_channel_paths.items(), channel_data):
        del label
        with tifffile.TiffWriter(str(path), append=True) as writer:
            writer.write(channel_frame.astype(np.float32))
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
    write_metadata(modality, str(error) if error is not None else None)
    if modality._save_enabled and modality._save_root_path is not None and modality._raw_frames:
        npz_path = modality._save_root_path.with_name(
            f"{modality._save_root_path.name}_raw.npz"
        )
        np.savez_compressed(
            npz_path,
            frames=np.asarray(modality._raw_frames, dtype=object),
            acquisition_parameters=np.asarray(modality._parameters_as_dict(), dtype=object),
        )


def write_metadata(modality, last_error: str | None) -> None:
    if not modality._save_enabled or modality._save_json_path is None:
        return
    payload = {
        "run_id": modality._run_id,
        "started": modality._run_started_at,
        "modality_key": modality.MODALITY_KEY,
        "save_root_path": str(modality._save_root_path),
        "tiff_paths": {label: str(path) for label, path in modality._save_channel_paths.items()},
        "frames_saved": modality._saved_frame_count,
        "frame_limit": modality._run_frame_limit,
        "parameters": modality._parameters_as_dict(),
        "last_error": last_error,
    }
    modality._save_json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
