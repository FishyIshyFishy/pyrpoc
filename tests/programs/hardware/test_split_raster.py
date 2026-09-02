"""Split confocal's device wiring and window split.

``reshape_to_split_frame`` is pinned by ``tests/reference/``. What is checked
here is the part that only runs on the instrument otherwise: the same
device-config-to-NI-arguments translation ``raster_scan`` does, plus the
``SplitGroup`` reaching the reshape.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.core.params import ScanGroup, SplitGroup
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.programs.hardware import split_raster

SCAN = ScanGroup(x_pixels=4, y_pixels=3, extra_left=1, extra_right=2, dwell_time_us=4.0)


@pytest.fixture
def devices():
    daq, galvo = DAQ(), Galvo()
    daq.config.device_name = "Dev7"
    daq.config.ai_channels = (2, 5)
    galvo.config.fast_ao = 3
    galvo.config.slow_ao = 4
    return daq, galvo


def fake_run_raster(captured, *, pixel_samples):
    def run(**kwargs):
        captured.update(kwargs)
        data = np.zeros((2, 3, 4 * pixel_samples), dtype=np.float32)
        return data, 3, 4, pixel_samples

    return run


def test_split_raster_scan_reads_its_wiring_off_the_devices(monkeypatch, devices):
    captured: dict = {}
    monkeypatch.setattr(split_raster, "run_raster", fake_run_raster(captured, pixel_samples=4))
    daq, galvo = devices

    split_raster.split_raster_scan(
        daq=daq,
        galvo=galvo,
        scan=SCAN,
        sample_rate_hz=1_000_000.0,
        split=SplitGroup(t0_samples=1, t1_samples=1),
        ttl={},
    )

    assert captured["device_name"] == "Dev7"
    assert captured["ai_channels"] == [2, 5]
    assert (captured["fast_ao"], captured["slow_ao"]) == (3, 4)
    assert (captured["x_pixels"], captured["y_pixels"]) == (4, 3)
    assert (captured["extra_left"], captured["extra_right"]) == (1, 2)
    assert captured["ttl_signals"] == {}


def test_the_split_group_reaches_the_reshape(monkeypatch, devices):
    """t0/t1 used to arrive as two splatted primitives; they arrive as the group."""
    captured: dict = {}
    monkeypatch.setattr(split_raster, "run_raster", fake_run_raster(captured, pixel_samples=4))
    seen: dict = {}

    def fake_reshape(scan_data, total_y, x_pixels, pixel_samples, t0_samples, t1_samples):
        seen.update(t0_samples=t0_samples, t1_samples=t1_samples)
        return np.zeros((4, total_y, x_pixels), np.float32), np.zeros((1, 1, 1, 1), np.float32)

    monkeypatch.setattr(split_raster, "reshape_to_split_frame", fake_reshape)
    daq, galvo = devices

    split_raster.split_raster_scan(
        daq=daq,
        galvo=galvo,
        scan=SCAN,
        sample_rate_hz=1_000_000.0,
        split=SplitGroup(t0_samples=2, t1_samples=1),
        ttl={},
    )
    assert seen == {"t0_samples": 2, "t1_samples": 1}


def test_split_raster_scan_requires_a_ttl_argument(devices):
    daq, galvo = devices
    with pytest.raises(TypeError):
        split_raster.split_raster_scan(  # type: ignore[call-arg]
            daq=daq,
            galvo=galvo,
            scan=SCAN,
            sample_rate_hz=1_000_000.0,
            split=SplitGroup(),
        )
