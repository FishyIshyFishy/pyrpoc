"""Freeze what v3.0's storage modules wrote, before phase 9 deletes them.

Phases 3 to 8 compared ``data/io.py`` against ``modalities/*/storage.py``
directly, which was the strongest check in the build. Phase 9 removes that
oracle. Rather than lose the comparison, this records the v3.0 output once and
the tests keep comparing against the recording.

Run against the v3.0 tree, and only ever deliberately -- like
``generate_references.py``, regenerating it on a changed implementation would
silently rebase the thing the tests compare against.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

reference_path = Path(__file__).parent / "storage_reference.npz"

STARTED_AT = "2026-08-30T00:00:00+00:00"
PARAMETERS = {"x_pixels": 4, "y_pixels": 3, "dwell_time_us": 2.0}
LABELS = ["ai0", "ai1"]


def frames(count: int, channels: int = 2, h: int = 3, w: int = 4) -> list[np.ndarray]:
    """Deterministic stand-in frames. Must match tests/data/test_io.py."""
    rng = np.random.default_rng(20260830)
    return [rng.normal(size=(channels, h, w)).astype(np.float32) for _ in range(count)]


class V30Modality:
    """The attribute surface modalities/*/storage.py reaches into."""

    modality_key = "confocal"

    def __init__(self, root: Path, labels: list[str]):
        self._save_enabled = True
        self._save_root_path = root
        self._save_json_path = None
        self._save_channel_paths = {}
        self._save_channel_labels = []
        self._saved_frame_count = 0
        self._run_id = 0
        self._run_started_at = ""
        self._run_frame_limit = None
        self._labels = labels
        self._pending_auxiliary = {}
        self._auxiliary_payload_buffers = {}
        self._auxiliary_paths = {}
        self._pending_flim_frame = None
        self._raw_frames = []

    def split_channels(self, data):
        return [data[i] for i in range(data.shape[0])]

    def resolve_channel_labels(self, count):
        return list(self._labels[:count])

    def parameters_as_dict(self):
        return dict(PARAMETERS)


def run_v30(storage, root: Path, labels, data, *, flim_cubes=None, split_raw=None):
    modality = V30Modality(root, labels)
    storage.prepare_acquisition_storage(modality, frame_limit=len(data))
    modality._run_started_at = STARTED_AT
    modality._run_id = 1
    for index, frame in enumerate(data):
        if flim_cubes is not None:
            modality._pending_flim_frame = flim_cubes[index]
        if split_raw is not None:
            modality._pending_auxiliary = {"raw_pixel_stream": split_raw[index]}
        storage.save_acquired_frame(modality, frame, frame_index=index)
    storage.finalize_acquisition_storage(
        modality, frame_count=len(data), frame_limit=len(data), error=None
    )
    return modality


def build_reference() -> dict[str, np.ndarray]:
    from pyrpoc.modalities.confocal import storage as confocal_storage
    from pyrpoc.modalities.flim import storage as flim_storage
    from pyrpoc.modalities.split_confocal import storage as split_storage

    out: dict[str, np.ndarray] = {}

    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "acq"
        data = frames(3)
        run_v30(confocal_storage, root, LABELS, data)
        for label in LABELS:
            out[f"confocal_tiff_{label}"] = np.frombuffer(
                root.with_name(f"acq_{label}.tiff").read_bytes(), dtype=np.uint8
            )
        meta = json.loads(root.with_name("acq_meta.json").read_text())
        out["confocal_meta"] = np.asarray(
            {key: meta[key] for key in
             ("run_id", "started", "modality_key", "frames_saved", "frame_limit",
              "parameters", "last_error")},
            dtype=object,
        )

    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "acq"
        data = frames(2, channels=4)
        raw = [
            np.random.default_rng(7).normal(size=(2, 3, 4, 5)).astype(np.float32)
            for _ in range(2)
        ]
        labels = [f"ai{n}_{w}" for n in (0, 1) for w in ("t0", "t2")]
        run_v30(split_storage, root, labels, data, split_raw=raw)
        with np.load(root.with_name("acq_raw_pixel_stream.npz"), allow_pickle=True) as npz:
            out["split_raw_frames"] = npz["frames"]
            out["split_raw_frame_indices"] = npz["frame_indices"]
        out["split_raw_keys"] = np.asarray(sorted(npz.files), dtype=object)

    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "acq"
        intensity = frames(2, channels=1)
        cubes = [
            np.random.default_rng(11).normal(size=(3, 4, 5)).astype(np.float32)
            for _ in range(2)
        ]
        run_v30(flim_storage, root, ["intensity"], intensity, flim_cubes=cubes)
        with np.load(root.with_name("acq_raw.npz"), allow_pickle=True) as npz:
            out["flim_cube_frames"] = np.stack(list(npz["frames"]), axis=0)
        out["flim_intensity_tiff"] = np.frombuffer(
            root.with_name("acq_intensity.tiff").read_bytes(), dtype=np.uint8
        )

    return out


def main() -> None:
    reference = build_reference()
    np.savez_compressed(reference_path, **reference)
    print(f"wrote {reference_path} ({len(reference)} entries)")
    for name, value in sorted(reference.items()):
        print(f"  {name:28s} {str(getattr(value, 'shape', '-')):16s} {value.dtype}")


if __name__ == "__main__":
    main()
