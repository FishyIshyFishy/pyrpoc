from __future__ import annotations

import dataclasses

import pytest

from pyrpoc.modalities.confocal.parameters import ConfocalParameters
from pyrpoc.modalities.split_confocal.parameters import SplitConfocalParameters


def confocal_raw(**overrides):
    raw = {
        "X Pixels": 256,
        "Y Pixels": 128,
        "Extra Steps Left": 10,
        "Extra Steps Right": 5,
        "Fast Axis Offset": 0.0,
        "Fast Axis Amplitude": 1.0,
        "Slow Axis Offset": 0.0,
        "Slow Axis Amplitude": 1.0,
        "Dwell Time (us)": 2.0,
        "DAQ Device": "Dev2",
        "Sample Rate (Hz)": 200000.0,
        "Fast Axis AO": 0,
        "Slow Axis AO": 1,
        "Active AI Channels": [0, 2, 4],
        "save_enabled": True,
        "save_path": "out/acq",
        "num_frames": 7,
    }
    raw.update(overrides)
    return raw


def test_confocal_from_dict_parses_fields():
    params = ConfocalParameters.from_dict(confocal_raw())
    assert params.x_pixels == 256
    assert params.y_pixels == 128
    assert params.device_name == "Dev2"
    assert params.active_ai_channels == (0, 2, 4)
    assert params.save_enabled is True
    assert params.num_frames == 7


def test_confocal_amplitude_is_clamped():
    params = ConfocalParameters.from_dict(confocal_raw(**{"Fast Axis Amplitude": 0.0}))
    assert params.fast_axis_amplitude == pytest.approx(1e-6)


def test_confocal_device_blank_falls_back_to_dev1():
    params = ConfocalParameters.from_dict(confocal_raw(**{"DAQ Device": ""}))
    assert params.device_name == "Dev1"


def test_confocal_is_frozen():
    params = ConfocalParameters.from_dict(confocal_raw())
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.x_pixels = 1  # type: ignore[misc]


def test_confocal_active_channels_is_tuple():
    params = ConfocalParameters.from_dict(confocal_raw())
    assert isinstance(params.active_ai_channels, tuple)


def split_raw(**overrides):
    raw = confocal_raw()
    raw.update({"t0 Samples": 3, "t1 Samples": 2})
    raw.update(overrides)
    return raw


def test_split_confocal_parses_timing():
    params = SplitConfocalParameters.from_dict(split_raw())
    assert params.t0_samples == 3
    assert params.t1_samples == 2


def test_split_confocal_rejects_t0_below_one():
    with pytest.raises(ValueError):
        SplitConfocalParameters.from_dict(split_raw(**{"t0 Samples": 0}))


def test_split_confocal_rejects_negative_t1():
    with pytest.raises(ValueError):
        SplitConfocalParameters.from_dict(split_raw(**{"t1 Samples": -1}))
