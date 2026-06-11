from __future__ import annotations

import pytest

from pyrpoc.domain.stores import ObjectStore


def test_add_returns_item_and_stores():
    store: ObjectStore[str] = ObjectStore([])
    assert store.add("a") == "a"
    assert store.list() == ["a"]


def test_list_returns_copy():
    store: ObjectStore[int] = ObjectStore([1, 2])
    snapshot = store.list()
    snapshot.append(3)
    assert store.list() == [1, 2]


def test_remove_present_and_absent():
    store: ObjectStore[str] = ObjectStore(["a", "b"])
    store.remove("a")
    assert store.list() == ["b"]
    # removing an absent item is a no-op, not an error
    store.remove("missing")
    assert store.list() == ["b"]


def test_contains():
    store: ObjectStore[str] = ObjectStore(["x"])
    assert store.contains("x")
    assert not store.contains("y")


def test_clear_returns_removed_items():
    store: ObjectStore[int] = ObjectStore([1, 2, 3])
    removed = store.clear()
    assert removed == [1, 2, 3]
    assert store.list() == []


def test_index_of():
    store: ObjectStore[str] = ObjectStore(["a", "b", "c"])
    assert store.index_of("b") == 1
    with pytest.raises(ValueError):
        store.index_of("z")
