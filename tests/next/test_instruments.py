import numpy as np
import pytest

from pyrpoc_next.instruments import (
    NIDAQ,
    PriorStage,
    RasterScan,
    SimulatedDAQ,
    SimulatedTagger,
    TimeTagger,
    instrument_registry,
    mask_to_ttl,
    raster_waveform,
)
from pyrpoc_next.structs.keys import InstrumentKey
from pyrpoc_next.structs.status import ConnectionStatus


def small_scan(**overrides):
    params = dict(x_pixels=8, y_pixels=6, dwell_time_us=2.0, sample_rate_hz=100_000.0, ai_channels=[0, 1])
    params.update(overrides)
    return RasterScan(**params)


def test_registry_holds_every_instrument():
    keys = set(instrument_registry.available())
    assert {
        InstrumentKey.ni_daq,
        InstrumentKey.time_tagger,
        InstrumentKey.prior_stage,
        InstrumentKey.zaber_stage,
        InstrumentKey.simulated_daq,
        InstrumentKey.simulated_tagger,
    } <= keys


def test_registry_creates_by_key():
    daq = instrument_registry.create(InstrumentKey.simulated_daq)
    assert isinstance(daq, SimulatedDAQ)


def test_raster_waveform_shape():
    scan = small_scan()
    waveform = raster_waveform(scan)
    assert waveform.shape == (2, scan.total_x * scan.y_pixels * scan.pixel_samples)


def test_mask_to_ttl_gates_to_active_samples():
    scan = small_scan()
    mask = np.ones((scan.y_pixels, scan.x_pixels), dtype=np.uint8)
    signals = mask_to_ttl(mask, "Dev1", 0, 0, scan, active_samples=1)
    (channel, ttl), = signals.items()
    assert channel == "Dev1/port0/line0"
    per_pixel = ttl.reshape(scan.y_pixels, scan.total_x, scan.pixel_samples)
    assert per_pixel[:, scan.extra_left : scan.extra_left + scan.x_pixels, 0].all()
    assert not per_pixel[:, :, 1:].any()


def test_mask_to_ttl_empty_mask_yields_nothing():
    scan = small_scan()
    mask = np.zeros((scan.y_pixels, scan.x_pixels), dtype=np.uint8)
    assert mask_to_ttl(mask, "Dev1", 0, 0, scan) == {}


def test_simulated_daq_returns_sample_cube_and_animates():
    daq = SimulatedDAQ()
    scan = small_scan()
    cube = daq.run(scan)
    assert cube.shape == (2, scan.y_pixels, scan.x_pixels, scan.pixel_samples)
    second = daq.run(scan)
    assert not np.array_equal(cube, second)  # frame index advances


def test_simulated_tagger_returns_histogram_cube():
    tagger = SimulatedTagger()
    cube = tagger.flim_frame(8, 6, 16, bin_width_ps=100.0, laser_period_ps=12500.0)
    assert cube.shape == (6, 8, 16)


def test_nidaq_run_without_hardware_raises_daq_error():
    from pyrpoc_next.instruments import DaqError

    with pytest.raises(DaqError):
        NIDAQ().run(small_scan())


def test_stage_placeholder_tracks_target():
    stage = PriorStage()
    stage.move_to(1.0, 2.0, 3.0)
    assert stage.position == (1.0, 2.0, 3.0)


def test_instrument_connect_sets_status():
    daq = SimulatedDAQ()
    assert daq.status is ConnectionStatus.untested
    daq.test_connection()
    assert daq.status is ConnectionStatus.ok
