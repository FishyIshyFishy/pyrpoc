"""Devices: configuration, identity, connection, persistence."""

from __future__ import annotations

import pytest

from pyrpoc.core.errors import TaggerError
from pyrpoc.devices import DAQ, Device, Galvo, TimeTagger, device_registry


# --- registry --------------------------------------------------------------


def test_all_three_devices_are_registered():
    assert device_registry.keys() == ["daq", "galvo", "time_tagger"]


def test_registry_records_the_key_on_the_class():
    assert DAQ.registry_key == "daq"
    assert device_registry.key_for(Galvo) == "galvo"


def test_registry_rejects_a_duplicate_key():
    with pytest.raises(KeyError, match="already registered"):
        device_registry.register("daq")(Galvo)


# --- configuration ---------------------------------------------------------


def test_daq_defaults_match_v3_0():
    daq = DAQ()
    assert daq.config.device_name == "Dev1"
    assert daq.config.ai_channels == tuple(range(9))


def test_galvo_defaults_match_v3_0():
    galvo = Galvo()
    assert galvo.config.fast_ao == 0
    assert galvo.config.slow_ao == 1


def test_tagger_defaults_match_v3_0():
    config = TimeTagger().config
    assert (config.laser_channel, config.detector_channel) == (1, 2)
    assert (config.pixel_channel, config.frame_channel) == (3, 4)
    assert config.laser_trigger_v == pytest.approx(0.05)
    assert config.detector_trigger_v == pytest.approx(0.2)
    assert config.pixel_trigger_v == pytest.approx(0.5)
    assert config.frame_trigger_v == pytest.approx(0.5)
    assert config.laser_input_delay_ps == 0


def test_device_config_unpacks_into_an_operation():
    """raster_scan(**daq.config, **galvo.config) must supply exactly its wiring args."""
    assert set(DAQ().config.keys()) == {"device_name", "ai_channels"}
    assert set(Galvo().config.keys()) == {"fast_ao", "slow_ao"}


# --- identity and claims ---------------------------------------------------


def test_galvo_is_backed_by_the_daq_and_owns_no_connection():
    assert Galvo.backed_by is DAQ
    assert Galvo.owns_connection is False
    assert DAQ.owns_connection is True
    assert DAQ.backed_by is None


def test_instance_ids_are_unique_and_prefixed():
    a, b = DAQ(), DAQ()
    assert a.instance_id != b.instance_id
    assert a.instance_id.startswith("daq-")


def test_user_label_overrides_the_display_name():
    assert DAQ().name == "NI-DAQ"
    assert DAQ(user_label="Upstairs card").name == "Upstairs card"


def test_summaries_describe_the_wiring():
    assert DAQ().summary().startswith("Dev1 - AI0")
    assert Galvo().summary() == "fast ao0, slow ao1"
    assert TimeTagger().summary() == "Connection: not tested"


# --- connection ------------------------------------------------------------


def test_test_connection_records_failure_without_raising(monkeypatch):
    daq = DAQ()
    monkeypatch.setattr(
        DAQ, "check_reachable", lambda self: (_ for _ in ()).throw(RuntimeError("no card"))
    )
    assert daq.test_connection() is False
    assert daq.last_test_ok is False
    assert daq.last_error == "no card"


def test_test_connection_records_success(monkeypatch):
    daq = DAQ()
    monkeypatch.setattr(DAQ, "check_reachable", lambda self: True)
    assert daq.test_connection() is True
    assert daq.last_test_ok is True
    assert daq.last_error is None


def test_a_device_without_a_connection_always_tests_ok():
    assert Galvo().test_connection() is True


def test_tagger_flim_calls_require_a_tagger():
    tagger = TimeTagger()
    with pytest.raises(TaggerError, match="create_tagger"):
        tagger.configure_for_flim()
    with pytest.raises(TaggerError, match="create_tagger"):
        tagger.start_flim_measurement(n_pixels=10, n_bins=5, binwidth_ps=100)


def test_stop_flim_measurement_is_safe_with_nothing_running():
    TimeTagger().stop_flim_measurement(None)


def test_configure_for_flim_writes_every_trigger_level():
    class FakeTagger:
        def __init__(self):
            self.levels = {}
            self.delays = {}

        def setTriggerLevel(self, channel, volts):
            self.levels[channel] = volts

        def setInputDelay(self, channel, ps):
            self.delays[channel] = ps

    device = TimeTagger()
    device.tagger = FakeTagger()
    device.config.laser_input_delay_ps = 250
    device.configure_for_flim()

    assert device.tagger.levels == {1: 0.05, 2: 0.2, 3: 0.5, 4: 0.5}
    assert device.tagger.delays == {1: 250}


def test_configure_for_flim_skips_a_zero_delay():
    class FakeTagger:
        def __init__(self):
            self.delays = {}

        def setTriggerLevel(self, channel, volts):
            pass

        def setInputDelay(self, channel, ps):
            self.delays[channel] = ps

    device = TimeTagger()
    device.tagger = FakeTagger()
    device.configure_for_flim()
    assert device.tagger.delays == {}


# --- persistence -----------------------------------------------------------


@pytest.mark.parametrize("cls", [DAQ, Galvo, TimeTagger])
def test_state_round_trip(cls):
    original = cls()
    if isinstance(original, DAQ):
        original.config.device_name = "Dev3"
        original.config.ai_channels = (0, 2)
    elif isinstance(original, Galvo):
        original.config.fast_ao = 4
    else:
        original.config.laser_channel = 7
    original.last_test_ok = True

    restored = cls()
    restored.import_state(original.export_state())

    assert restored.export_state() == original.export_state()
    assert restored.last_test_ok is True


def test_import_state_tolerates_junk():
    daq = DAQ()
    daq.import_state({})
    daq.import_state(None)  # type: ignore[arg-type]
    assert daq.config.device_name == "Dev1"


def test_base_panel_is_none_so_a_generated_form_is_the_whole_panel():
    assert Galvo().panel() is None
    assert Device().panel() is None
