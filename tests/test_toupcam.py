"""The ToupTek camera's hardware-free parts: frame unpacking, error naming,
camera choice, dropping the dark frame after a start, and the greyscale
reduction Brightfield publishes."""

from __future__ import annotations

import threading
import time
from typing import cast

import numpy as np
import pytest

from pyrpoc.plugins.devices.touptek_camera.device import ToupcamCamera
from pyrpoc.plugins.devices.touptek_camera.sdk import (
    CameraInfo,
    ToupcamError,
    ToupcamSession,
    describe_hresult,
    find_camera,
    row_pitch,
    unpack_rgb24,
)
from pyrpoc.plugins.programs.brightfield.program import to_grey
from pyrpoc.structs.data_library.data import Image2D


def test_row_pitch_pads_to_four_bytes():
    assert row_pitch(4) == 12
    assert row_pitch(5) == 16
    assert row_pitch(3328) == 3328 * 3


def test_unpack_drops_row_padding():
    width, height = 5, 2
    pitch = row_pitch(width)
    buffer = np.full(pitch * height, 255, dtype=np.uint8)
    for row in range(height):
        buffer[row * pitch : row * pitch + width * 3] = np.arange(width * 3) + 10 * row
    frame = unpack_rgb24(buffer, width, height)
    assert frame.shape == (height, width, 3)
    assert frame[1, 0, 0] == 10
    assert 255 not in frame


def test_negative_hresult_reads_as_a_cause():
    assert describe_hresult(-2147024891) == "access denied"  # 0x80070005 as a signed long
    assert describe_hresult(-1) == "error 0xFFFFFFFF"


def test_find_camera_by_name_or_first():
    cameras = [CameraInfo("A", "id-a", ()), CameraInfo("B", "id-b", ())]
    assert find_camera(cameras, "").name == "A"
    assert find_camera(cameras, "B").id == "id-b"
    assert find_camera(cameras, "id-a").name == "A"
    with pytest.raises(ToupcamError):
        find_camera(cameras, "C")
    with pytest.raises(ToupcamError):
        find_camera([], "")


def test_grey_is_a_one_channel_image():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[..., 0] = 30
    grey = to_grey(frame)
    assert grey.shape == (1, 4, 6)
    assert grey.dtype == np.float32
    assert np.allclose(grey, 10.0)
    Image2D.validate(grey)


class FakeSession:
    """Delivers numbered 2x2 frames, each filled with its number, one per pull."""

    def __init__(self):
        self.frame_ready = threading.Event()
        self.frame_ready.set()
        self.fault = None
        self.pulled = 0

    def exposure_range_us(self):
        return 100, 2_000_000

    def gain_range_pct(self):
        return 100, 300

    def start(self, size_index, exposure_us, gain_pct):
        pass

    def frame_size(self):
        return 2, 2

    def pull(self, buffer):
        buffer[:] = self.pulled
        self.pulled += 1
        self.frame_ready.set()
        return 2, 2


def test_first_frame_after_start_is_dropped():
    camera = ToupcamCamera()
    camera.session = cast(ToupcamSession, FakeSession())
    camera.start(exposure_us=10_000, gain_pct=100, binned=False)
    assert camera.grab(time.sleep)[0, 0, 0] == 1
    assert camera.grab(time.sleep)[0, 0, 0] == 2
    camera.start(exposure_us=10_000, gain_pct=100, binned=False)
    assert camera.grab(time.sleep)[0, 0, 0] == 4
