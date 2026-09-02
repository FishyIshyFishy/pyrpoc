"""Datasets: appending, reading, and change notification."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from pyrpoc.core.streams import Cube3D, Image2D
from pyrpoc.data.dataset import Dataset, Provenance


def make(spec=Image2D, **kwargs) -> Dataset:
    return Dataset(
        stream=kwargs.pop("stream", "intensity"),
        spec=spec,
        provenance=kwargs.pop("provenance", Provenance("confocal", run_id=1)),
        **kwargs,
    )


def test_append_returns_increasing_indices():
    ds = make()
    assert [ds.append(np.zeros((1, 2, 2), np.float32)) for _ in range(3)] == [0, 1, 2]
    assert len(ds) == 3


def test_append_validates_against_the_contract():
    ds = make()
    with pytest.raises(ValueError, match="3 dimensions"):
        ds.append(np.zeros((2, 2), np.float32))
    assert len(ds) == 0


def test_append_casts_to_float32():
    ds = make()
    ds.append(np.zeros((1, 2, 2), dtype=np.float64))
    assert ds.frame(0).dtype == np.float32


def test_latest_and_frame_return_what_was_appended():
    ds = make()
    first = np.full((1, 2, 2), 1.0, np.float32)
    second = np.full((1, 2, 2), 2.0, np.float32)
    ds.append(first)
    ds.append(second)
    np.testing.assert_array_equal(ds.frame(0), first)
    np.testing.assert_array_equal(ds.latest(), second)


def test_latest_is_none_before_anything_arrives():
    assert make().latest() is None
    assert make().stack() is None


def test_stack_adds_a_leading_frame_axis():
    ds = make()
    for _ in range(3):
        ds.append(np.zeros((2, 4, 5), np.float32))
    assert ds.stack().shape == (3, 2, 4, 5)


def test_channel_labels_are_filled_in_on_first_append():
    ds = make()
    ds.append(np.zeros((3, 2, 2), np.float32))
    assert ds.channel_labels == ["channel_0", "channel_1", "channel_2"]


def test_supplied_channel_labels_are_kept():
    ds = make(channel_labels=["ai0", "ai1"])
    ds.append(np.zeros((2, 2, 2), np.float32))
    assert ds.channel_labels == ["ai0", "ai1"]


def test_cube_datasets_get_no_channel_labels():
    """Cube3D's first axis is y, not channel."""
    ds = make(spec=Cube3D, stream="histogram")
    ds.append(np.zeros((4, 5, 6), np.float32))
    assert ds.channel_labels == []


def test_coords_are_recorded_per_frame():
    ds = make()
    ds.append(np.zeros((1, 2, 2), np.float32), z=1.5)
    assert ds.coords(0) == {"z": 1.5}


# --- change notification ---------------------------------------------------


def test_subscribers_fire_once_per_append():
    ds = make()
    seen = []
    ds.subscribe(lambda dataset, index: seen.append(index))
    for _ in range(3):
        ds.append(np.zeros((1, 2, 2), np.float32))
    assert seen == [0, 1, 2]


def test_unsubscribe_stops_notification():
    ds = make()
    seen = []

    def callback(dataset, index):
        seen.append(index)

    ds.subscribe(callback)
    ds.append(np.zeros((1, 2, 2), np.float32))
    ds.unsubscribe(callback)
    ds.append(np.zeros((1, 2, 2), np.float32))
    assert seen == [0]


def test_subscribing_twice_still_notifies_once():
    ds = make()
    seen = []

    def callback(dataset, index):
        seen.append(index)

    ds.subscribe(callback)
    ds.subscribe(callback)
    ds.append(np.zeros((1, 2, 2), np.float32))
    assert seen == [0]


def test_notification_carries_the_dataset_itself():
    ds = make()
    seen = []
    ds.subscribe(lambda dataset, index: seen.append(dataset))
    ds.append(np.zeros((1, 2, 2), np.float32))
    assert seen == [ds]


def test_appending_from_another_thread_is_safe():
    """publish() runs on the worker thread; reads happen on the GUI thread."""
    ds = make()

    def append_many():
        for _ in range(50):
            ds.append(np.zeros((1, 2, 2), np.float32))

    threads = [threading.Thread(target=append_many) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(ds) == 200


# --- identity --------------------------------------------------------------


def test_ids_are_unique_and_carry_the_stream_name():
    a, b = make(), make()
    assert a.id != b.id
    assert a.id.startswith("intensity-")


def test_label_names_the_program_run_and_stream():
    label = make().label
    assert "confocal" in label and "#1" in label and "intensity" in label


def test_run_id_comes_from_provenance():
    assert make(provenance=Provenance("flim", run_id=7)).run_id == 7


def test_finalize_without_a_writer_is_a_no_op():
    make().finalize(0, None)
