"""Saving, checked against what v3.0's storage modules wrote.

Phases 3 to 8 ran the old code and the new code side by side and compared the
results file by file. Phase 9 deleted the old code, so the comparison now runs
against ``tests/reference/storage_reference.npz`` -- a recording of that same
output, frozen while both implementations were still in the tree.

If one of these fails, the files this application writes have changed. Fix the
implementation; do not regenerate the reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pyrpoc.core.errors import ParameterError
from pyrpoc.core.streams import Cube3D, Image2D, Samples4D
from pyrpoc.data.dataset import Dataset, Provenance
from pyrpoc.data.io import RunSaver, SaveTarget, load_frames

from tests.reference.generate_storage_reference import reference_path


@pytest.fixture(scope="module")
def v30():
    """What v3.0's storage modules wrote for these inputs."""
    if not reference_path.exists():
        pytest.fail(f"missing {reference_path}")
    with np.load(reference_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


STARTED_AT = "2026-08-30T00:00:00+00:00"
PARAMETERS = {"x_pixels": 4, "y_pixels": 3, "dwell_time_us": 2.0}


def frames(count: int, channels: int = 2, h: int = 3, w: int = 4) -> list[np.ndarray]:
    rng = np.random.default_rng(20260830)
    return [
        rng.normal(size=(channels, h, w)).astype(np.float32) for _ in range(count)
    ]


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


def test_tiff_output_matches_v30_byte_for_byte(tmp_path, v30):
    data = frames(3)
    labels = ["ai0", "ai1"]
    root = tmp_path / "acq"
    run_new(root, {"intensity": Image2D}, {"intensity": data}, labels)

    for label in labels:
        written = np.frombuffer(
            root.with_name(f"acq_{label}.tiff").read_bytes(), dtype=np.uint8
        )
        np.testing.assert_array_equal(
            written,
            v30[f"confocal_tiff_{label}"],
            err_msg=f"acq_{label}.tiff differs from the v3.0 output",
        )


def test_tiff_holds_one_page_per_frame_in_order(tmp_path):
    data = frames(3)
    root = tmp_path / "acq"
    run_new(root, {"intensity": Image2D}, {"intensity": data}, ["ai0", "ai1"])

    pages = load_frames(root.with_name("acq_ai0.tiff"))
    assert pages.shape == (3, 3, 4)
    for index, frame in enumerate(data):
        np.testing.assert_array_equal(pages[index], frame[0])


def test_metadata_keys_v30_wrote_have_the_same_values(tmp_path, v30):
    root = tmp_path / "acq"
    run_new(root, {"intensity": Image2D}, {"intensity": frames(3)}, ["ai0", "ai1"])
    meta = json.loads(root.with_name("acq_meta.json").read_text())

    for key, expected in v30["confocal_meta"].item().items():
        assert meta[key] == expected, f"metadata key {key!r} changed from the v3.0 value"
    assert sorted(meta["tiff_paths"]) == ["ai0", "ai1"]
    assert Path(meta["save_root_path"]).name == "acq"


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


def test_split_raw_npz_matches_v30(tmp_path, v30):
    data = frames(2, channels=4)
    raw = [np.random.default_rng(7).normal(size=(2, 3, 4, 5)).astype(np.float32) for _ in range(2)]
    labels = [f"ai{n}_{w}" for n in (0, 1) for w in ("t0", "t2")]

    root = tmp_path / "acq"
    run_new(
        root,
        {"intensity": Image2D, "raw_pixel_stream": Samples4D},
        {"intensity": data, "raw_pixel_stream": raw},
        labels,
    )

    written = root.with_name("acq_raw_pixel_stream.npz")
    assert written.exists(), "the filename must be unchanged from v3.0"
    with np.load(written, allow_pickle=True) as new:
        assert sorted(new.files) == list(v30["split_raw_keys"])
        np.testing.assert_array_equal(new["frames"], v30["split_raw_frames"])
        np.testing.assert_array_equal(new["frame_indices"], v30["split_raw_frame_indices"])
        assert new["frames"].dtype == np.float32


# --- Cube3D: FLIM's histogram ----------------------------------------------


def test_flim_histogram_npz_holds_the_same_cubes_as_v30(tmp_path, v30):
    intensity = frames(2, channels=1)
    cubes = [np.random.default_rng(11).normal(size=(3, 4, 5)).astype(np.float32) for _ in range(2)]

    root = tmp_path / "acq"
    run_new(
        root,
        {"intensity": Image2D, "histogram": Cube3D},
        {"intensity": intensity, "histogram": cubes},
        ["intensity"],
    )

    expected = np.asarray(v30["flim_cube_frames"].tolist(), dtype=np.float32)
    with np.load(root.with_name("acq_histogram.npz"), allow_pickle=True) as new:
        np.testing.assert_array_equal(new["frames"], expected)
        # The recorded improvements: a real array, and a uniform key name.
        assert new["frames"].dtype == np.float32
        assert "parameters" in new.files

    written = np.frombuffer(root.with_name("acq_intensity.tiff").read_bytes(), dtype=np.uint8)
    np.testing.assert_array_equal(written, v30["flim_intensity_tiff"])


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


# --- the save target --------------------------------------------------------


def test_the_root_is_the_directory_and_the_name():
    target = SaveTarget(name="run1", directory="/data", enabled=True)
    assert target.root == Path("/data/run1")


def test_a_name_is_a_filename_not_a_path():
    """The field is labelled Name and the folder is picked separately."""
    assert SaveTarget(name="/elsewhere/run1", directory="/data").root == Path("/data/run1")


def test_a_tiff_suffix_is_stripped():
    """The writers append ``_<channel>.tiff``; a typed one would land inside it."""
    assert SaveTarget(name="run1.tiff", directory="/data").root == Path("/data/run1")
    assert SaveTarget(name="run1.TIF", directory="/data").filename == "run1"


def test_no_directory_means_the_working_directory():
    assert SaveTarget(name="run1").root == Path.cwd() / "run1"


def test_a_home_relative_directory_is_expanded():
    target = SaveTarget(name="run1", directory="~/data")
    assert target.root == Path.home() / "data" / "run1"


def test_a_blank_name_is_an_error_only_when_a_root_is_asked_for():
    """Saving off with no name is a normal state -- most of a session is that."""
    assert SaveTarget(name="   ").filename == ""
    with pytest.raises(ParameterError, match="Name is required"):
        _ = SaveTarget(name="   ", enabled=True).root


def test_the_folder_is_separate_from_the_root_so_the_ui_can_name_it():
    assert SaveTarget(directory="/data", name="run1").folder == Path("/data")
    assert SaveTarget(name="run1").folder == Path.cwd()
