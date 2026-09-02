"""Capturing and applying a session against live shell objects."""

from __future__ import annotations

import pytest

from pyrpoc.session.store import SessionStore
from pyrpoc.shell.app import Application
from pyrpoc.shell.session_io import Autosave, apply, capture, seed_defaults
from pyrpoc.shell.theme.manager import ThemeController


@pytest.fixture
def app(qapp):
    application = Application()
    application.add_device("daq")
    application.add_device("galvo")
    application.select_program("confocal")
    return application


@pytest.fixture
def theme(qapp):
    return ThemeController(qapp)


# --- capture ---------------------------------------------------------------


def test_capture_records_devices_with_their_configuration(app):
    app.devices[0].config.device_name = "Dev7"
    state = capture(app)
    assert [row.key for row in state.devices] == ["daq", "galvo"]
    assert state.devices[0].state["config"]["device_name"] == "Dev7"


def test_capture_records_the_selected_program_and_its_parameters(app):
    app.current_params().scan.x_pixels = 64
    state = capture(app)
    assert state.selected_program == "confocal"
    assert state.params_by_program["confocal"]["scan"]["x_pixels"] == 64


def test_capture_records_views(app):
    from pyrpoc.views.image_2d import Image2DView

    view = Image2DView()
    view.user_label = "Channels"
    app.add_view(view)

    state = capture(app)
    assert [row.key for row in state.views] == ["image_2d"]
    assert state.views[0].user_label == "Channels"


# --- apply -----------------------------------------------------------------


def test_apply_rebuilds_devices_and_their_configuration(app):
    app.devices[0].config.device_name = "Dev7"
    app.devices[1].config.fast_ao = 3
    state = capture(app)

    restored = Application()
    apply(state, restored)

    assert [type(d).__name__ for d in restored.devices] == ["DAQ", "Galvo"]
    assert restored.devices[0].config.device_name == "Dev7"
    assert restored.devices[1].config.fast_ao == 3


def test_apply_restores_parameters_per_program(app):
    app.params_for("confocal").scan.x_pixels = 64
    app.params_for("flim").scan.y_pixels = 256
    state = capture(app)

    restored = Application()
    apply(state, restored)
    assert restored.params_for("confocal").scan.x_pixels == 64
    assert restored.params_for("flim").scan.y_pixels == 256


def test_apply_restores_views_and_their_state(app):
    from pyrpoc.views.image_2d import Image2DView

    view = Image2DView()
    view.user_label = "Channels"
    app.add_view(view)
    state = capture(app)

    restored = Application()
    apply(state, restored)
    assert len(restored.views) == 1
    assert restored.views[0].user_label == "Channels"


def test_apply_skips_a_device_type_that_no_longer_exists(app):
    from pyrpoc.session.state import DeviceState

    state = capture(app)
    state.devices.append(DeviceState("prior_stage", "gone-1", None, {}))

    restored = Application()
    apply(state, restored)
    assert [type(d).__name__ for d in restored.devices] == ["DAQ", "Galvo"]


def test_apply_falls_back_when_the_saved_program_is_gone(app):
    state = capture(app)
    state.selected_program = "z_stack_flim"

    restored = Application()
    apply(state, restored)
    assert restored.selected_program == "confocal"


def test_apply_clears_what_was_there_before(app):
    state = capture(app)
    other = Application()
    other.add_device("time_tagger")
    apply(state, other)
    assert [type(d).__name__ for d in other.devices] == ["DAQ", "Galvo"]


# --- defaults ---------------------------------------------------------------


def test_seed_defaults_gives_a_fresh_workbench_a_card_and_a_galvo(qapp):
    """Without this the schema-7 reset leaves a dead play button.

    v3.0 confocal needed no instruments; v3.1 declares uses = [Galvo, DAQ].
    """
    fresh = Application()
    seed_defaults(fresh)
    assert [type(d).__name__ for d in fresh.devices] == ["DAQ", "Galvo"]
    fresh.select_program("confocal")
    assert fresh.missing_devices("confocal") == []


def test_seed_defaults_leaves_an_existing_workbench_alone(app):
    seed_defaults(app)
    assert len(app.devices) == 2


# --- autosave ---------------------------------------------------------------


def test_autosave_writes_on_demand(app, theme, tmp_path):
    store = SessionStore(tmp_path / "session.json")
    autosave = Autosave(app, None, theme, store)
    autosave.save_now()
    assert store.path.exists()
    assert store.load().selected_program == "confocal"


def test_restore_reads_back_a_saved_session(app, theme, tmp_path):
    store = SessionStore(tmp_path / "session.json")
    app.devices[0].config.device_name = "Dev9"
    Autosave(app, None, theme, store).save_now()

    fresh = Application()
    Autosave(fresh, None, theme, store).restore()
    assert fresh.devices[0].config.device_name == "Dev9"


def test_restore_seeds_defaults_when_there_is_no_session(theme, tmp_path):
    fresh = Application()
    Autosave(fresh, None, theme, SessionStore(tmp_path / "none.json")).restore()
    assert [type(d).__name__ for d in fresh.devices] == ["DAQ", "Galvo"]


def test_reset_clears_and_reseeds(app, theme, tmp_path):
    app.params_for("confocal").scan.x_pixels = 64
    autosave = Autosave(app, None, theme, SessionStore(tmp_path / "session.json"))
    autosave.reset()
    assert [type(d).__name__ for d in app.devices] == ["DAQ", "Galvo"]
    assert app.params_for("confocal").scan.x_pixels == 512


def test_autosave_is_suspended_during_a_restore(app, theme, tmp_path):
    """Restoring fires devices_changed repeatedly; none of it should be saved."""
    store = SessionStore(tmp_path / "session.json")
    autosave = Autosave(app, None, theme, store)
    saves = []
    autosave.save_now = lambda: saves.append(1)  # type: ignore[method-assign]
    autosave.suspended = True
    autosave.schedule()
    assert saves == []


def test_a_failed_save_never_interrupts_an_experiment(app, theme, tmp_path):
    class Exploding(SessionStore):
        def save(self, state):
            raise OSError("disk full")

    Autosave(app, None, theme, Exploding(tmp_path / "session.json")).save_now()


# --- the save target --------------------------------------------------------


def test_capture_records_the_save_target(app):
    app.set_save(name="cells", directory="/data", enabled=True)
    state = capture(app)
    assert (state.save.name, state.save.directory, state.save.enabled) == (
        "cells",
        "/data",
        True,
    )


def test_apply_restores_the_save_target(app):
    app.set_save(name="cells", directory="/data", enabled=True)
    state = capture(app)

    restored = Application()
    apply(state, restored)
    assert restored.save.name == "cells"
    assert restored.save.directory == "/data"
    assert restored.save.enabled is True


def test_apply_tolerates_a_session_saved_before_saving_moved_out(app):
    """The save group used to sit inside each program's parameters.

    A v7 file still carries one there. It is no longer a field, so it is
    dropped rather than blocking the restore -- which is why the schema was
    not bumped over this.
    """
    state = capture(app)
    state.params_by_program["confocal"]["save"] = {"save_enabled": True, "save_path": "/old"}
    state.params_by_program["confocal"]["scan"]["x_pixels"] = 64

    restored = Application()
    apply(state, restored)
    assert restored.params_for("confocal").scan.x_pixels == 64
    assert not hasattr(restored.params_for("confocal"), "save")
    assert restored.save.enabled is False


def test_reset_clears_the_save_target_too(app, theme, tmp_path):
    app.set_save(name="cells", directory="/data", enabled=True)
    Autosave(app, None, theme, SessionStore(tmp_path / "session.json")).reset()
    assert (app.save.name, app.save.directory, app.save.enabled) == ("acquisition", "", False)
