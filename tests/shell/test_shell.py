"""The shell: state, the launcher, the panels, and a run end to end.

Headless via QT_QPA_PLATFORM=offscreen (see tests/conftest.py).
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from pyrpoc.core.errors import MissingDevice
from pyrpoc.core.streams import Image2D
from pyrpoc.shell import catalog
from pyrpoc.shell.app import Application
from pyrpoc.shell.display_bridge import DisplayBridge
from pyrpoc.shell.launcher import LauncherPanel
from pyrpoc.shell.param_form import ParamForm


def pump(seconds: float = 2.0, until=None) -> None:
    """Run the Qt event loop until a condition holds or time runs out."""
    deadline = threading.Event()
    threading.Timer(seconds, deadline.set).start()
    while not deadline.is_set():
        QApplication.processEvents()
        if until is not None and until():
            return
    QApplication.processEvents()


@pytest.fixture
def app(qapp):
    application = Application()
    application.add_device("daq")
    application.add_device("galvo")
    return application


def small_scan(params) -> None:
    params.scan.x_pixels = 8
    params.scan.y_pixels = 8
    params.scan.extra_left = 3
    params.scan.extra_right = 2
    params.num_frames = 2


# --- catalog ---------------------------------------------------------------


def test_the_catalog_offers_the_three_programs():
    assert catalog.keys() == ["confocal", "split_confocal", "flim"]


def test_the_catalog_carries_labels_not_registry_keys():
    """v3.0's dropdown showed `split_confocal`."""
    assert [e.label for e in catalog.CATALOG] == ["Confocal", "Split Confocal", "FLIM"]


def test_an_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        catalog.entry_for("nope")


# --- application state -----------------------------------------------------


def test_selecting_a_program_creates_its_parameter_model(app):
    app.select_program("confocal")
    from pyrpoc.programs.confocal import ConfocalParams

    assert isinstance(app.current_params(), ConfocalParams)


def test_each_program_keeps_its_own_parameters(app):
    app.select_program("confocal")
    app.current_params().scan.x_pixels = 64
    app.select_program("flim")
    app.current_params().scan.x_pixels = 128
    app.select_program("confocal")
    assert app.current_params().scan.x_pixels == 64


def test_missing_devices_are_named_per_program(app):
    app.clear_devices()
    assert set(app.missing_devices("confocal")) == {"Galvo", "NI-DAQ"}
    app.add_device("daq")
    app.add_device("galvo")
    assert app.missing_devices("confocal") == []
    assert app.missing_devices("flim") == ["Swabian TimeTagger"]


def test_devices_can_be_added_and_removed(app):
    assert [type(d).__name__ for d in app.devices] == ["DAQ", "Galvo"]
    app.remove_device(app.devices[0])
    assert [type(d).__name__ for d in app.devices] == ["Galvo"]


def test_parameter_state_round_trips(app):
    app.select_program("confocal")
    app.current_params().scan.x_pixels = 64
    raw = app.params_state()

    restored = Application()
    restored.load_params_state(raw)
    assert restored.params_for("confocal").scan.x_pixels == 64


def test_loading_parameter_state_ignores_unknown_programs(app):
    app.load_params_state({"gone": {"scan": {"x_pixels": 32}}})
    assert "gone" not in app.params_by_program


def test_loading_corrupt_parameter_state_falls_back_to_defaults(app):
    app.load_params_state({"confocal": {"scan": {"x_pixels": "not a number"}}})
    assert app.params_for("confocal").scan.x_pixels == 512


# --- the generated form ----------------------------------------------------


def test_the_form_has_a_section_per_group(qapp):
    from pyrpoc.programs.confocal import ConfocalParams

    params = ConfocalParams()
    form = ParamForm(params)
    assert sorted(form.fields) [:3] == ["daq.sample_rate_hz", "modulation.masks", "num_frames"]
    assert "scan.x_pixels" in form.fields


def test_editing_a_widget_writes_straight_into_the_model(qapp):
    """Settled statement 3: the model is authoritative, not the form."""
    from pyrpoc.programs.confocal import ConfocalParams

    params = ConfocalParams()
    form = ParamForm(params)
    form.fields["scan.x_pixels"].set(64)
    form.on_field_changed()
    assert params.scan.x_pixels == 64


def test_the_form_loads_from_the_model(qapp):
    from pyrpoc.programs.confocal import ConfocalParams

    params = ConfocalParams()
    params.scan.y_pixels = 128
    form = ParamForm(params)
    assert form.fields["scan.y_pixels"].get() == 128


def test_every_field_type_round_trips_through_its_widget(qapp):
    from pathlib import Path

    from pyrpoc.core.modulation import MaskBinding
    from pyrpoc.programs.flim import FlimParams
    from pyrpoc.programs.split_confocal import SplitConfocalParams

    for cls in (SplitConfocalParams, FlimParams):
        params = cls()
        form = ParamForm(params)
        form.read_into()

    params = SplitConfocalParams()
    form = ParamForm(params)
    form.fields["modulation.masks"].set((MaskBinding(Path("m.png"), 1, 2),))
    form.on_field_changed()
    assert params.modulation.masks == (MaskBinding(Path("m.png"), 1, 2),)

    form.fields["save.save_enabled"].set(True)
    form.fields["save.save_path"].set("/tmp/run")
    form.on_field_changed()
    assert params.save.save_enabled is True
    assert params.save.save_path == "/tmp/run"


def test_the_channel_row_reads_back_as_a_tuple(qapp):
    from pyrpoc.devices import DAQ

    daq = DAQ()
    form = ParamForm(daq.config, cards=False)
    form.fields["ai_channels"].set((0, 2, 5))
    form.on_field_changed()
    assert daq.config.ai_channels == (0, 2, 5)


def test_an_out_of_bounds_value_is_reported_not_raised(qapp):
    """An exception escaping a Qt slot aborts the process, so it must not.

    The spin boxes clamp, so this needs the minimum lowered to reach at all --
    but a form that can kill the application when it does is not acceptable.
    """
    from pyrpoc.programs.confocal import ConfocalParams

    params = ConfocalParams()
    form = ParamForm(params)
    reported = []
    form.invalid.connect(reported.append)

    form.fields["scan.x_pixels"].widget.setMinimum(0)
    form.fields["scan.x_pixels"].set(2)
    form.on_field_changed()

    assert reported and "must be >= 8" in reported[0]
    assert params.scan.x_pixels == 512, "the model must not take the bad value"


def test_read_into_still_raises_for_programmatic_callers(qapp):
    from pyrpoc.core.errors import ParameterError
    from pyrpoc.programs.confocal import ConfocalParams

    form = ParamForm(ConfocalParams())
    form.fields["scan.x_pixels"].widget.setMinimum(0)
    form.fields["scan.x_pixels"].set(2)
    with pytest.raises(ParameterError):
        form.read_into()


# --- the launcher ----------------------------------------------------------


def test_the_launcher_lists_programs_by_label(app):
    panel = LauncherPanel(app)
    assert [panel.program_combo.itemText(i) for i in range(panel.program_combo.count())] == [
        "Confocal",
        "Split Confocal",
        "FLIM",
    ]


def test_choosing_a_program_rebuilds_the_form(app):
    panel = LauncherPanel(app)
    panel.program_combo.setCurrentIndex(panel.program_combo.findData("flim"))
    assert app.selected_program == "flim"
    assert "histogram.histogram_bins" in panel.form.fields


def test_missing_devices_disable_play_and_say_what_is_needed(app):
    panel = LauncherPanel(app)
    panel.program_combo.setCurrentIndex(panel.program_combo.findData("flim"))
    assert panel.start_btn.isEnabled() is False
    assert "Swabian TimeTagger" in panel.status_label.text()

    app.add_device("time_tagger")
    assert panel.start_btn.isEnabled() is True
    assert panel.status_label.text() == "Status: ready"


def test_starting_with_no_devices_reports_instead_of_crashing(app, monkeypatch):
    app.clear_devices()
    app.select_program("confocal")
    monkeypatch.setattr(
        "pyrpoc.shell.launcher.QMessageBox.critical", lambda *args, **kwargs: None
    )
    panel = LauncherPanel(app)
    panel.start(continuous=False)
    assert "Galvo" in panel.status_label.text()


# --- a run, end to end ------------------------------------------------------


def test_a_run_publishes_into_the_library_and_reaches_the_launcher(app, monkeypatch):
    monkeypatch.setattr(
        "pyrpoc.programs.confocal.raster_scan",
        lambda **kwargs: np.zeros((2, 8, 8), np.float32),
    )
    app.select_program("confocal")
    small_scan(app.current_params())
    app.devices[0].config.ai_channels = (0, 1)

    panel = LauncherPanel(app)
    statuses = []
    app.bridge.run_status.connect(statuses.append)

    handle = app.start_run()
    pump(until=lambda: not handle.thread.is_alive())
    pump(0.2)

    assert len(handle.datasets["intensity"]) == 2
    assert app.library.matching(Image2D) == [handle.datasets["intensity"]]
    assert statuses == ["frame 1/2", "frame 2/2"]
    assert "stopped" in panel.status_label.text()


def test_dataset_changes_arrive_on_the_gui_thread(app, monkeypatch):
    """run/ is Qt-free; RunBridge is what hops the thread boundary."""
    monkeypatch.setattr(
        "pyrpoc.programs.confocal.raster_scan",
        lambda **kwargs: np.zeros((2, 8, 8), np.float32),
    )
    app.select_program("confocal")
    small_scan(app.current_params())
    app.devices[0].config.ai_channels = (0, 1)

    gui_thread = threading.current_thread()
    seen_on: list[threading.Thread] = []
    app.bridge.dataset_changed.connect(lambda *_: seen_on.append(threading.current_thread()))

    handle = app.start_run()
    pump(until=lambda: len(seen_on) >= 2)
    handle.thread.join(timeout=5)

    assert len(seen_on) == 2
    assert all(thread is gui_thread for thread in seen_on)


def test_the_display_bridge_renders_into_a_v30_display(app, monkeypatch):
    """Temporary, deleted in phase 8, but it must work while it is here."""
    from pyrpoc.displays.tiled_2d_display import Tiled2DDisplay

    monkeypatch.setattr(
        "pyrpoc.programs.confocal.raster_scan",
        lambda **kwargs: np.full((2, 8, 8), 7.0, np.float32),
    )
    DisplayBridge(app, app)
    display = Tiled2DDisplay()
    display.configure({})
    app.add_view(display)

    app.select_program("confocal")
    small_scan(app.current_params())
    app.devices[0].config.ai_channels = (0, 1)

    handle = app.start_run()
    pump(until=lambda: not handle.thread.is_alive())
    pump(0.2)

    np.testing.assert_array_equal(display._data_chw, np.full((2, 8, 8), 7.0, np.float32))


def test_a_saved_run_writes_the_expected_files(app, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "pyrpoc.programs.confocal.raster_scan",
        lambda **kwargs: np.zeros((2, 8, 8), np.float32),
    )
    app.select_program("confocal")
    params = app.current_params()
    small_scan(params)
    params.save.save_enabled = True
    params.save.save_path = str(tmp_path / "acq")
    app.devices[0].config.ai_channels = (0, 1)

    handle = app.start_run()
    pump(until=lambda: not handle.thread.is_alive())

    assert (tmp_path / "acq_ai0.tiff").exists()
    meta = json.loads((tmp_path / "acq_meta.json").read_text())
    assert meta["program_key"] == "confocal"
    assert meta["frames_saved"] == 2


def test_continuous_runs_until_stopped(app, monkeypatch):
    monkeypatch.setattr(
        "pyrpoc.programs.confocal.raster_scan",
        lambda **kwargs: np.zeros((2, 8, 8), np.float32),
    )
    app.select_program("confocal")
    params = app.current_params()
    small_scan(params)
    params.num_frames = 1
    app.devices[0].config.ai_channels = (0, 1)

    handle = app.start_run(continuous=True)
    pump(until=lambda: len(handle.datasets["intensity"]) > 3)
    app.stop_run()
    handle.thread.join(timeout=5)

    assert len(handle.datasets["intensity"]) > 1
    assert params.num_frames == 1


def test_starting_a_run_with_no_program_selected_is_an_error(app):
    app.selected_program = None
    with pytest.raises(RuntimeError, match="no program selected"):
        app.start_run()


def test_a_run_without_the_devices_raises_before_anything_starts(app):
    app.clear_devices()
    app.select_program("confocal")
    with pytest.raises(MissingDevice):
        app.start_run()
    assert len(app.library) == 0
