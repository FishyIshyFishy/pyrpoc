"""Session persistence at schema 7."""

from __future__ import annotations

import json

import pytest

from pyrpoc.session.state import SCHEMA_VERSION, DeviceState, SessionState, ViewState
from pyrpoc.session.store import SessionStore, decode, default_session_path


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session.json")


def sample() -> SessionState:
    return SessionState(
        theme_mode="dark",
        devices=[
            DeviceState("daq", "daq-1", "Upstairs card", {"config": {"device_name": "Dev3"}}),
            DeviceState("galvo", "galvo-1", None, {"config": {"fast_ao": 2}}),
        ],
        views=[ViewState("tiled_2d", "view-1", "Channels", True, {"channels": []})],
        selected_program="flim",
        params_by_program={"confocal": {"scan": {"x_pixels": 64}}},
        ads_layout="Zm9v",
    )


# --- round trip ------------------------------------------------------------


def test_save_then_load_round_trips(store):
    store.save(sample())
    loaded = store.load()
    assert loaded == sample()
    assert store.last_load_error is None


def test_a_missing_file_gives_defaults(store):
    loaded = store.load()
    assert loaded == SessionState()
    assert store.last_load_error is None


def test_the_file_is_readable_json(store):
    store.save(sample())
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["devices"][0]["key"] == "daq"


def test_saving_creates_the_directory(tmp_path):
    store = SessionStore(tmp_path / "nested" / "deeper" / "session.json")
    store.save(SessionState())
    assert store.path.exists()


def test_the_write_is_atomic(store):
    """A crash mid-write must not leave a half-written session."""
    store.save(sample())
    assert not store.path.with_suffix(".json.tmp").exists()


# --- the schema 7 reset ----------------------------------------------------


def test_a_v6_session_loads_as_defaults_without_an_error(store):
    """The recorded one-time reset. v6 stored parameters by widget label and
    instruments rather than devices, so there is nothing to map onto."""
    legacy = {
        "schema_version": 6,
        "theme_mode": "dark",
        "instruments": [{"type_key": "time_tagger"}],
        "modality": {"selected_key": "confocal", "params_by_modality": {}},
    }
    store.path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = store.load()
    assert loaded == SessionState()
    assert store.last_load_error is None, "a version reset is expected, not an error"


def test_a_future_schema_also_resets(store):
    store.path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    assert store.load() == SessionState()


# --- damaged files ----------------------------------------------------------


def test_corrupt_json_reports_and_falls_back(store):
    store.path.write_text("{not json", encoding="utf-8")
    loaded = store.load()
    assert loaded == SessionState()
    assert store.last_load_error and "could not read" in store.last_load_error


def test_a_json_scalar_is_not_a_session(store):
    store.path.write_text("42", encoding="utf-8")
    assert store.load() == SessionState()
    assert store.last_load_error and "does not contain a session" in store.last_load_error


def test_rows_without_a_key_are_skipped():
    state = decode(
        {
            "schema_version": SCHEMA_VERSION,
            "devices": [{"key": "daq"}, {"instance_id": "orphan"}, "junk"],
            "views": [{"key": "tiled_2d"}, {}],
        }
    )
    assert [row.key for row in state.devices] == ["daq"]
    assert [row.key for row in state.views] == ["tiled_2d"]


def test_params_that_are_not_objects_are_skipped():
    state = decode(
        {
            "schema_version": SCHEMA_VERSION,
            "params_by_program": {"confocal": {"num_frames": 2}, "broken": "nope"},
        }
    )
    assert list(state.params_by_program) == ["confocal"]


def test_a_non_string_layout_is_dropped():
    assert decode({"schema_version": SCHEMA_VERSION, "ads_layout": 7}).ads_layout is None


# --- the default location ---------------------------------------------------


def test_the_default_path_is_under_a_config_directory():
    path = default_session_path()
    assert path.name == "session.json"
    assert path.parent.name == "pyrpoc"


def test_session_state_knows_when_it_is_empty():
    assert SessionState().is_empty() is True
    assert sample().is_empty() is False
