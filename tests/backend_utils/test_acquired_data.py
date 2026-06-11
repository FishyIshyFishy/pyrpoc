from __future__ import annotations

import numpy as np

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.backend_utils.contracts import Action
from pyrpoc.backend_utils.opto_control_contexts import BaseOptoControlContext, MaskContext


def test_datakind_values_are_stable_strings():
    assert DataKind.INTENSITY_FRAME.value == "intensity_frame"
    assert DataKind.PARTIAL_FRAME.value == "partial_frame"
    assert DataKind.FLIM_RAW_FRAME.value == "flim_raw_frame"


def test_only_intensity_frame_is_persistent():
    assert DataKind.INTENSITY_FRAME.is_persistent
    assert not DataKind.PARTIAL_FRAME.is_persistent
    assert not DataKind.FLIM_RAW_FRAME.is_persistent
    assert not DataKind.FLIM_PARTIAL_HISTOGRAM.is_persistent


def test_acquired_data_defaults():
    data = AcquiredData(data=np.zeros((2, 2), dtype=np.float32), kind=DataKind.INTENSITY_FRAME)
    assert data.channel_labels == []
    assert data.metadata == {}


def test_acquired_data_independent_default_containers():
    a = AcquiredData(data=np.zeros((1, 1), dtype=np.float32), kind=DataKind.PARTIAL_FRAME)
    b = AcquiredData(data=np.zeros((1, 1), dtype=np.float32), kind=DataKind.PARTIAL_FRAME)
    a.channel_labels.append("ch0")
    a.metadata["k"] = 1
    assert b.channel_labels == []
    assert b.metadata == {}


def test_action_defaults():
    action = Action(label="Run", method_name="run")
    assert action.parameters == []
    assert action.tooltip == ""
    assert action.dangerous is False
    assert action.confirm_text is None


def test_action_independent_parameter_lists():
    a = Action(label="A", method_name="a")
    a.parameters.append("x")  # type: ignore[arg-type]
    b = Action(label="B", method_name="b")
    assert b.parameters == []


def test_mask_context_is_opto_context():
    ctx = MaskContext(optocontrol_key="mask", alias="m1", mask=None, daq_port=1, daq_line=2)
    assert isinstance(ctx, BaseOptoControlContext)
    assert ctx.daq_port == 1
    assert ctx.daq_line == 2
