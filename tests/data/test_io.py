"""Saving, checked against the v3.0 storage modules that are still in the tree.

This is the strongest check available in this build and it disappears at phase
9, so it is written properly: the old code and the new code save the same frames
to two directories and the results are compared file by file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pyrpoc.core.streams import Cube3D, Image2D, Samples4D
from pyrpoc.data.dataset import Dataset, Provenance
from pyrpoc.data.io import RunSaver, load_frames

# The v3.0 implementations, still present until phase 9.
from pyrpoc.modalities.confocal import storage as v30_confocal_storage
from pyrpoc.modalities.flim import storage as v30_flim_storage
from pyrpoc.modalities.split_confocal import storage as v30_split_storage


STARTED_AT = "2026-08-30T00:00:00+00:00"
PARAMETERS = {"x_pixels": 4, "y_pixels": 3, "dwell_time_us": 2.0}


def frames(count: int, channels: int = 2, h: int = 3, w: int = 4) -> list[np.ndarray]:
    rng = np.random.default_rng(20260830)
    return [
        rng.normal(size=(channels, h, w)).astype(np.float32) for _ in range(count)
    ]


class V30Modality:
    """The attribute surface modalities/*/storage.py reaches into."""

    modality_key = "confocal"

    def __init__(self, root: Path, labels: list[str], enabled: bool = True):
        self._save_enabled = enabled
        self._save_root_path = root
        self._save_json_path = None
        self._save_channel_paths = {}
        self._save_channel_labels = []
        self._saved_frame_count = 0
        self._run_id = 0
        self._run_started_at = ""
        self._run_frame_limit = None
        self._labels = labels
        # split confocal only
        self._pending_auxiliary = {}
        self._auxiliary_payload_buffers = {}
        self._auxiliary_paths = {}
        # flim only
        self._pending_flim_frame = None
        self._raw_frames = []

    def split_channels(self, data):
        return [data[i] for i in range(data.shape[0])]

    def resolve_channel_labels(self, count):
        return list(self._labels[:count])

    def parameters_as_dict(self):
        return dict(PARAMETERS)


def run_v30(storage, root: Path, labels: list[str], data: list[np.ndarray], **extra):
    modality = V30Modality(root, labels)
    for key, value in extra.items():
        setattr(modality, key, value)
    storage.prepare_acquisition_storage(modality, frame_limit=len(data))
    modality._run_started_at = STARTED_AT
    modality._run_id = 1
    for index, frame in enumerate(data):
        if extra.get("_flim_cubes") is not None:
            modality._pending_flim_frame = extra["_flim_cubes"][index]
        if extra.get("_split_raw") is not None:
            modality._pending_auxiliary = {"raw_pixel_stream": extra["_split_raw"][index]}
        storage.save_acquired_frame(modality, frame, frame_index=index)
    storage.finalize_acquisition_storage(
        modality, frame_count=len(data), frame_limit=len(data), error=None
    )
    return modality


def run_new(root: Path, streams: dict, published: dict[str, list[np.ndarray]], labels):
    saver = RunSaver(
        root=root,
        program_key="confocal",
        parameters=dict(PARAMETERS),
        run_id=1,
        started_at=STARTED_AT,
        frame_limit=len(next(iter(published.values()))),
    )
    saver.prepare(streams)
    provenance = Provenance("confocal", dict(PARAMETERS), {}, STARTED_AT, 1)
    datasets = {
        stream: Dataset(
            stream=stream,
            spec=spec,
            provenance=provenance,
            channel_labels=labels if spec is Image2D else None,
            writer=saver.writer_for(stream),
        )
        for stream, spec in streams.items()
    }
    count = len(next(iter(published.values())))
    for index in range(count):
        for stream, arrays in published.items():
            datasets[stream].append(arrays[index])
    for dataset in datasets.values():
        dataset.finalize(count, None)
    saver.finalize(count, None)
    return saver, datasets


# --- Image2D: the confocal path -------------------------------------------


def test_tiff_output_matches_v30_byte_for_byte(tmp_path):
    data = frames(3)
    labels = ["ai0", "ai1"]

    old_root = tmp_path / "old" / "acq"
    old_root.parent.mkdir(parents=True)
    run_v30(v30_confocal_storage, old_root, labels, data)

    new_root = tmp_path / "new" / "acq"
    run_new(new_root, {"intensity": Image2D}, {"intensity": data}, labels)

    for label in labels:
        old_bytes = (old_root.with_name(f"acq_{label}.tiff")).read_bytes()
        new_bytes = (new_root.with_name(f"acq_{label}.tiff")).read_bytes()
        assert old_bytes == new_bytes, f"{label}.tiff differs from the v3.0 output"


def test_tiff_holds_one_page_per_frame_in_order(tmp_path):
    data = frames(3)
    root = tmp_path / "acq"
    run_new(root, {"intensity": Image2D}, {"intensity": data}, ["ai0", "ai1"])

    pages = load_frames(root.with_name("acq_ai0.tiff"))
    assert pages.shape == (3, 3, 4)
    for index, frame in enumerate(data):
        np.testing.assert_array_equal(pages[index], frame[0])


def test_metadata_keys_v30_wrote_have_the_same_values(tmp_path):
    data = frames(2)
    labels = ["ai0", "ai1"]

    old_root = tmp_path / "old" / "acq"
    old_root.parent.mkdir(parents=True)
    run_v30(v30_confocal_storage, old_root, labels, data)
    old_meta = json.loads(old_root.with_name("acq_meta.json").read_text())

    new_root = tmp_path / "new" / "acq"
    run_new(new_root, {"intensity": Image2D}, {"intensity": data}, labels)
    new_meta = json.loads(new_root.with_name("acq_meta.json").read_text())

    for key in ("run_id", "started", "modality_key", "frames_saved", "frame_limit",
                "parameters", "last_error"):
        assert new_meta[key] == old_meta[key], f"metadata key {key!r} changed"

    # Paths differ only by the temp directory they were written into.
    assert sorted(new_meta["tiff_paths"]) == sorted(old_meta["tiff_paths"])
    assert Path(new_meta["save_root_path"]).name == Path(old_meta["save_root_path"]).name


def test_metadata_adds_program_key_and_streams(tmp_path):
    root = tmp_path / "acq"
    run_new(root, {"intensity": Image2D}, {"intensity": frames(1)}, ["ai0", "ai1"])
    meta = json.loads(root.with_name("acq_meta.json").read_text())
    assert meta["program_key"] == "confocal"
    assert meta["modality_key"] == meta["program_key"]  # v3.0 alias
    assert meta["streams"] == ["intensity"]


def test_metadata_is_rewritten_after_every_frame(tmp_path):
    """Run progress must stay readable from disk mid-run, as it was in v3.0."""
    root = tmp_path / "acq"
    saver = RunSaver(root=root, program_key="confocal", parameters={}, started_at=STARTED_AT)
    saver.prepare({"intensity": Image2D})
    dataset = Dataset(
        stream="intensity",
        spec=Image2D,
        provenance=Provenance("confocal"),
        channel_labels=["ai0"],
        writer=saver.writer_for("intensity"),
    )
    seen = []
    for frame in frames(3, channels=1):
        dataset.append(frame)
        seen.append(json.loads(root.with_name("acq_meta.json").read_text())["frames_saved"])
    assert seen == [1, 2, 3]


def test_existing_tiffs_are_replaced_not_appended_to(tmp_path):
    root = tmp_path / "acq"
    run_new(root, {"intensity": Image2D}, {"intensity": frames(3, channels=1)}, ["ai0"])
    run_new(root, {"intensity": Image2D}, {"intensity": frames(2, channels=1)}, ["ai0"])
    pages = load_frames(root.with_name("acq_ai0.tiff"))
    assert pages.shape[0] == 2


def test_channel_count_mismatch_is_an_error(tmp_path):
    root = tmp_path / "acq"
    saver = RunSaver(root=root, program_key="confocal", parameters={})
    saver.prepare({"intensity": Image2D})
    dataset = Dataset(
        stream="intensity", spec=Image2D, provenance=Provenance("confocal"),
        channel_labels=["ai0", "ai1"], writer=saver.writer_for("intensity"),
    )
    dataset.append(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="channel count"):
        dataset.append(np.zeros((3, 3, 4), dtype=np.float32))


def test_unlabelled_channels_fall_back_to_channel_n(tmp_path):
    root = tmp_path / "acq"
    run_new(root, {"intensity": Image2D}, {"intensity": frames(1)}, None)
    assert root.with_name("acq_channel_0.tiff").exists()
    assert root.with_name("acq_channel_1.tiff").exists()


# --- Samples4D: split confocal's raw stream --------------------------------


def test_split_raw_npz_matches_v30(tmp_path):
    data = frames(2, channels=4)
    raw = [np.random.default_rng(7).normal(size=(2, 3, 4, 5)).astype(np.float32) for _ in range(2)]
    labels = [f"ai{n}_{w}" for n in (0, 1) for w in ("t0", "t2")]

    old_root = tmp_path / "old" / "acq"
    old_root.parent.mkdir(parents=True)
    run_v30(v30_split_storage, old_root, labels, data, _split_raw=raw)

    new_root = tmp_path / "new" / "acq"
    run_new(
        new_root,
        {"intensity": Image2D, "raw_pixel_stream": Samples4D},
        {"intensity": data, "raw_pixel_stream": raw},
        labels,
    )

    old_name = old_root.with_name("acq_raw_pixel_stream.npz")
    new_name = new_root.with_name("acq_raw_pixel_stream.npz")
    assert old_name.exists() and new_name.exists(), "the filename must be unchanged from v3.0"

    with np.load(old_name, allow_pickle=True) as old, np.load(new_name, allow_pickle=True) as new:
        assert sorted(old.files) == sorted(new.files) == ["frame_indices", "frames", "parameters"]
        np.testing.assert_array_equal(old["frames"], new["frames"])
        np.testing.assert_array_equal(old["frame_indices"], new["frame_indices"])
        assert old["frames"].dtype == new["frames"].dtype == np.float32


# --- Cube3D: FLIM's histogram ----------------------------------------------


def test_flim_histogram_npz_holds_the_same_cubes_as_v30(tmp_path):
    intensity = frames(2, channels=1)
    cubes = [np.random.default_rng(11).normal(size=(3, 4, 5)).astype(np.float32) for _ in range(2)]

    old_root = tmp_path / "old" / "acq"
    old_root.parent.mkdir(parents=True)
    run_v30(v30_flim_storage, old_root, ["intensity"], intensity, _flim_cubes=cubes)

    new_root = tmp_path / "new" / "acq"
    run_new(
        new_root,
        {"intensity": Image2D, "histogram": Cube3D},
        {"intensity": intensity, "histogram": cubes},
        ["intensity"],
    )

    with np.load(old_root.with_name("acq_raw.npz"), allow_pickle=True) as old:
        old_frames = np.stack(list(old["frames"]), axis=0)
    with np.load(new_root.with_name("acq_histogram.npz"), allow_pickle=True) as new:
        np.testing.assert_array_equal(new["frames"], old_frames)
        # The recorded improvements: a real array, and a uniform key name.
        assert new["frames"].dtype == np.float32
        assert "parameters" in new.files


def test_flim_histogram_filename_changed_deliberately(tmp_path):
    root = tmp_path / "acq"
    run_new(
        root,
        {"intensity": Image2D, "histogram": Cube3D},
        {"intensity": frames(1, channels=1), "histogram": [np.zeros((3, 4, 5), np.float32)]},
        ["intensity"],
    )
    assert root.with_name("acq_histogram.npz").exists()
    assert not root.with_name("acq_raw.npz").exists()


def test_npz_paths_appear_in_the_metadata(tmp_path):
    root = tmp_path / "acq"
    run_new(
        root,
        {"intensity": Image2D, "histogram": Cube3D},
        {"intensity": frames(1, channels=1), "histogram": [np.zeros((3, 4, 5), np.float32)]},
        ["intensity"],
    )
    meta = json.loads(root.with_name("acq_meta.json").read_text())
    assert "histogram" in meta["auxiliary_paths"]
    assert meta["streams"] == ["histogram", "intensity"]


# --- saving off ------------------------------------------------------------


def test_a_dataset_with_no_writer_writes_nothing(tmp_path):
    dataset = Dataset(stream="intensity", spec=Image2D, provenance=Provenance("confocal"))
    dataset.append(np.zeros((1, 3, 4), dtype=np.float32))
    dataset.finalize(1, None)
    assert list(tmp_path.iterdir()) == []
    assert len(dataset) == 1


def test_finalize_records_the_error(tmp_path):
    root = tmp_path / "acq"
    saver = RunSaver(root=root, program_key="confocal", parameters={})
    saver.prepare({"intensity": Image2D})
    saver.finalize(0, RuntimeError("NI-DAQ acquisition failed: boom"))
    meta = json.loads(root.with_name("acq_meta.json").read_text())
    assert "NI-DAQ acquisition failed" in meta["last_error"]


def test_a_plain_imread_returns_only_the_first_frame(tmp_path):
    """Documents the v3.0 on-disk quirk load_frames exists to work around.

    Appending page by page puts every frame in its own TIFF series, so
    ``tifffile.imread(path)`` gives frame 0 and nothing else. Preserved because
    changing it would change the files analysis scripts already read.
    """
    root = tmp_path / "acq"
    data = frames(3, channels=1)
    run_new(root, {"intensity": Image2D}, {"intensity": data}, ["ai0"])
    path = root.with_name("acq_ai0.tiff")

    np.testing.assert_array_equal(tifffile.imread(str(path)), data[0][0])
    assert load_frames(path).shape == (3, 3, 4)
