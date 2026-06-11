from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.rpoc.types import RPOCEditorState, RPOCImageInput, RPOCRoi


def test_from_array_promotes_2d_to_single_channel():
    rpoc = RPOCImageInput.from_array(np.zeros((8, 10), dtype=np.float32))
    assert rpoc.data.shape == (1, 8, 10)
    assert rpoc.channel_labels == ["Channel 1"]


def test_from_array_keeps_channels_first():
    rpoc = RPOCImageInput.from_array(np.zeros((3, 8, 10), dtype=np.float32))
    assert rpoc.data.shape == (3, 8, 10)


def test_from_array_moves_channels_last_to_first():
    # shape[0] > 8 forces the channels-last interpretation (C = trailing dim <= 8).
    rpoc = RPOCImageInput.from_array(np.zeros((10, 12, 3), dtype=np.float32))
    assert rpoc.data.shape == (3, 10, 12)


def test_from_array_ambiguous_3d_raises():
    with pytest.raises(ValueError):
        RPOCImageInput.from_array(np.zeros((16, 16, 16), dtype=np.float32))


def test_from_array_rejects_4d():
    with pytest.raises(ValueError):
        RPOCImageInput.from_array(np.zeros((2, 3, 4, 5), dtype=np.float32))


def test_from_array_casts_to_float32():
    rpoc = RPOCImageInput.from_array(np.ones((4, 4), dtype=np.int16))
    assert rpoc.data.dtype == np.float32


def test_from_array_source_id_passthrough():
    rpoc = RPOCImageInput.from_array(np.zeros((4, 4), dtype=np.float32), source_id="disp-1")
    assert rpoc.source_id == "disp-1"


def test_direct_construction_requires_3d():
    with pytest.raises(ValueError):
        RPOCImageInput(data=np.zeros((4, 4), dtype=np.float32))


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError):
        RPOCImageInput(data=np.zeros((2, 4, 4), dtype=np.float32), channel_labels=["only one"])


def test_default_labels_match_channel_count():
    rpoc = RPOCImageInput(data=np.zeros((3, 4, 4), dtype=np.float32))
    assert rpoc.channel_labels == ["Channel 1", "Channel 2", "Channel 3"]


def test_roi_defaults():
    roi = RPOCRoi(roi_id=1, points=[(0.0, 0.0)], threshold_low=0.1, threshold_high=0.9)
    assert roi.modulation_level == 0.5
    assert roi.active_channels == []


def test_editor_state_defaults():
    state = RPOCEditorState()
    assert state.image_input is None
    assert state.rois == []
    assert state.show_rois is True
    assert state.next_roi_id == 1
