from __future__ import annotations

from pathlib import Path

from pyrpoc.backend_utils.state_helpers import (
    export_object_state,
    import_object_state,
    make_instance_id,
    to_persistable,
)


# --------------------------------------------------------------------------- #
# make_instance_id
# --------------------------------------------------------------------------- #

def test_make_instance_id_format_and_uniqueness():
    a = make_instance_id("Mask Control")
    b = make_instance_id("Mask Control")
    assert a != b
    prefix, _, suffix = a.partition("-")
    assert prefix == "mask_control"
    assert len(suffix) == 12


def test_make_instance_id_blank_prefix_falls_back():
    assert make_instance_id("   ").startswith("item-")


# --------------------------------------------------------------------------- #
# to_persistable
# --------------------------------------------------------------------------- #

def test_to_persistable_primitives_and_path():
    assert to_persistable("x") == (True, "x")
    assert to_persistable(3) == (True, 3)
    assert to_persistable(None) == (True, None)
    ok, value = to_persistable(Path("a/b"))
    assert ok and isinstance(value, Path)


def test_to_persistable_tuple_becomes_list():
    assert to_persistable((1, 2, 3)) == (True, [1, 2, 3])


def test_to_persistable_dict_keys_stringified():
    ok, value = to_persistable({1: "a"})
    assert ok and value == {"1": "a"}


def test_to_persistable_rejects_unsupported_object():
    assert to_persistable(object()) == (False, None)


def test_to_persistable_depth_limit():
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}
    ok, _ = to_persistable(deep)
    assert ok is False


# --------------------------------------------------------------------------- #
# export / import object state
# --------------------------------------------------------------------------- #

class Sample:
    def __init__(self):
        self.name = "abc"
        self.count = 3
        self.path = Path("out")
        self._private = "skip"
        self.handler = lambda: None
        self.bad = object()


def test_export_skips_private_callable_and_unpersistable():
    state = export_object_state(Sample())
    assert state == {"name": "abc", "count": 3, "path": Path("out")}


def test_export_include_fields_filter():
    state = export_object_state(Sample(), include_fields={"name"})
    assert state == {"name": "abc"}


def test_export_exclude_fields_filter():
    state = export_object_state(Sample(), exclude_fields={"count", "path"})
    assert state == {"name": "abc"}


def test_import_sets_known_attributes():
    obj = Sample()
    import_object_state(obj, {"name": "zzz", "count": 9})
    assert obj.name == "zzz"
    assert obj.count == 9


def test_import_skips_unknown_when_no_include():
    obj = Sample()
    import_object_state(obj, {"unknown_attr": 1})
    assert not hasattr(obj, "unknown_attr")


def test_import_respects_exclude():
    obj = Sample()
    import_object_state(obj, {"name": "new"}, exclude_fields={"name"})
    assert obj.name == "abc"
