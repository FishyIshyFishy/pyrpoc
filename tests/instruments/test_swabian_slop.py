from __future__ import annotations

import numpy as np

from pyrpoc.instruments.swabian_slop import (
    FLIMConfig,
    process_photon_lifetimes,
    reshape_to_frame,
)


def make_config(**overrides) -> FLIMConfig:
    kwargs = dict(
        x_pixels=2,
        y_pixels=2,
        extra_left=1,
        extra_right=1,
        pixel_dwell_ps=100,
        laser_ch=1,
        detector_ch=2,
        pixel_ch=3,
    )
    kwargs.update(overrides)
    return FLIMConfig(**kwargs)


def test_flim_config_geometry_properties():
    config = make_config(x_pixels=2, y_pixels=2, extra_left=1, extra_right=1, pixel_dwell_ps=100)
    assert config.total_x_pixels == 4
    assert config.total_pixels == 8
    assert config.frame_duration_ps == 800


def test_process_photon_lifetimes_basic():
    timestamps = np.array([0, 100, 1000, 1100], dtype=np.int64)
    channels = np.array([1, 2, 1, 2], dtype=np.int64)  # laser, photon, laser, photon
    lifetimes, pixel_indices, last_laser = process_photon_lifetimes(
        timestamps, channels, make_config(), last_laser_ps=None
    )
    assert list(lifetimes) == [100, 100]
    assert list(pixel_indices) == [1, 11]
    assert last_laser == 1000


def test_process_photon_lifetimes_uses_carryover_laser():
    timestamps = np.array([50], dtype=np.int64)
    channels = np.array([2], dtype=np.int64)  # a lone detector event
    lifetimes, _, last_laser = process_photon_lifetimes(
        timestamps, channels, make_config(), last_laser_ps=10
    )
    assert list(lifetimes) == [40]  # 50 - carried-over laser at 10
    assert last_laser == 10  # no laser in this chunk, carryover preserved


def test_process_photon_lifetimes_no_detector_events():
    timestamps = np.array([0, 1000], dtype=np.int64)
    channels = np.array([1, 1], dtype=np.int64)  # only laser pulses
    lifetimes, pixel_indices, last_laser = process_photon_lifetimes(
        timestamps, channels, make_config(), last_laser_ps=None
    )
    assert lifetimes.size == 0
    assert pixel_indices.size == 0
    assert last_laser is None


def test_reshape_to_frame_groups_and_clips():
    config = make_config(x_pixels=2, y_pixels=2, extra_left=1, extra_right=1)
    lifetimes = np.array([10, 20, 30], dtype=np.int64)
    pixel_indices = np.array([1, 1, 5], dtype=np.int64)
    frame = reshape_to_frame(lifetimes, pixel_indices, config)

    assert frame.shape == (2, 2)  # y_pixels x x_pixels after clipping the overscan
    assert frame.dtype == object
    assert np.array_equal(frame[0, 0], np.array([10, 20], dtype=np.int64))
    assert np.array_equal(frame[1, 0], np.array([30], dtype=np.int64))
    assert frame[0, 1].size == 0


def test_reshape_to_frame_drops_out_of_bounds():
    config = make_config(x_pixels=2, y_pixels=2, extra_left=1, extra_right=1)
    lifetimes = np.array([10, 20], dtype=np.int64)
    pixel_indices = np.array([-1, 999], dtype=np.int64)
    frame = reshape_to_frame(lifetimes, pixel_indices, config)
    assert all(frame[y, x].size == 0 for y in range(2) for x in range(2))
