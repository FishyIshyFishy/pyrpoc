"""The CCD readout block's mapping to the device, and the device's checks on
values that come from the user."""

from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.plugins.devices import AndorError, SpectrumReadout
from pyrpoc.plugins.devices.andor_ccd.device import check_track, clipped, match_index
from pyrpoc.plugins.programs.building_blocks.parameter_groups import CcdReadoutGroup


def test_raman_readout_is_fvb_on_the_conventional_amplifier():
    group = CcdReadoutGroup(exposure_s=0.5, frames_to_average=4)
    assert group.to_readout() == SpectrumReadout(
        exposure_s=0.5,
        accumulations=4,
        read_mode="fvb",
        track_centre=1,
        track_height=1,
        amplifier="conventional",
        em_gain=1,
        readout_mhz=0.05,
        preamp_gain=1.0,
    )


def test_rate_matches_the_cameras_float32_table():
    # The Newton reports its rates as float32, so 0.05 arrives slightly off.
    table = [3.0, 1.0, float(np.float32(0.05))]
    assert match_index(table, 0.05, "readout rate (MHz)") == 2


def test_missing_rate_names_what_the_camera_offers():
    with pytest.raises(AndorError, match="offers"):
        match_index([3.0, 1.0], 0.05, "readout rate (MHz)")


def test_track_must_fit_on_the_sensor():
    check_track(100, 20, 200)
    with pytest.raises(AndorError):
        check_track(195, 20, 200)
    with pytest.raises(AndorError):
        check_track(5, 20, 200)


def test_clipping_scales_with_accumulations():
    counts = np.array([300, 65535], dtype=np.int32)
    assert clipped(counts, 1)
    assert not clipped(counts, 2)
