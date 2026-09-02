"""The library of open runs."""

from __future__ import annotations

from pyrpoc.core.streams import Cube3D, Image2D, Samples4D
from pyrpoc.data.dataset import Dataset, Provenance
from pyrpoc.data.library import DatasetLibrary


def make(stream="intensity", spec=Image2D, run_id=1) -> Dataset:
    return Dataset(stream=stream, spec=spec, provenance=Provenance("confocal", run_id=run_id))


def test_add_and_remove():
    library = DatasetLibrary()
    dataset = library.add(make())
    assert library.all() == [dataset] and len(library) == 1
    library.remove(dataset)
    assert library.all() == []


def test_removing_something_absent_is_a_no_op():
    library = DatasetLibrary()
    library.remove(make())
    assert len(library) == 0


def test_get_by_run_and_stream():
    library = DatasetLibrary()
    intensity = library.add(make("intensity", Image2D, run_id=2))
    library.add(make("histogram", Cube3D, run_id=2))
    assert library.get(2, "intensity") is intensity
    assert library.get(9, "intensity") is None


def test_by_id():
    library = DatasetLibrary()
    dataset = library.add(make())
    assert library.by_id(dataset.id) is dataset
    assert library.by_id("nope") is None


def test_matching_returns_newest_first():
    library = DatasetLibrary()
    first = library.add(make("intensity", Image2D, run_id=1))
    second = library.add(make("intensity", Image2D, run_id=2))
    library.add(make("raw_pixel_stream", Samples4D, run_id=2))
    assert library.matching(Image2D) == [second, first]


def test_matching_accepts_several_contracts():
    library = DatasetLibrary()
    library.add(make("intensity", Image2D))
    library.add(make("histogram", Cube3D))
    assert len(library.matching(Image2D, Cube3D)) == 2
    assert library.matching(Samples4D) == []


def test_subscribers_fire_on_membership_change():
    library = DatasetLibrary()
    calls = []
    library.subscribe(lambda: calls.append(1))
    dataset = library.add(make())
    library.remove(dataset)
    library.clear()
    assert len(calls) == 3


def test_unsubscribe():
    library = DatasetLibrary()
    calls = []

    def callback():
        calls.append(1)

    library.subscribe(callback)
    library.add(make())
    library.unsubscribe(callback)
    library.add(make())
    assert len(calls) == 1


def test_all_returns_a_copy():
    library = DatasetLibrary()
    library.add(make())
    library.all().clear()
    assert len(library) == 1


def test_datasets_outlive_whatever_rendered_them():
    """The point of the library: closing a view must not destroy the data."""
    library = DatasetLibrary()
    dataset = library.add(make())
    dataset.append(__import__("numpy").zeros((1, 2, 2), dtype="float32"))
    viewer = object()
    del viewer
    assert len(library.by_id(dataset.id)) == 1
