from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pyrpoc.instruments.time_tagger import TimeTaggerInstrument


def test_configure_for_flim_sets_all_trigger_levels():
    inst = TimeTaggerInstrument()
    inst.tagger = MagicMock()
    inst.configure_for_flim(
        1, 2, 3, 4,
        0.05, 0.2, 0.5, 0.6,
    )
    levels = {call.args[0]: call.args[1] for call in inst.tagger.setTriggerLevel.call_args_list}
    assert levels == {1: 0.05, 2: 0.2, 3: 0.5, 4: 0.6}
    inst.tagger.setInputDelay.assert_not_called()  # zero delay -> skipped


def test_configure_for_flim_applies_laser_input_delay():
    inst = TimeTaggerInstrument()
    inst.tagger = MagicMock()
    inst.configure_for_flim(1, 2, 3, 4, 0.05, 0.2, 0.5, 0.6, laser_input_delay_ps=-12500)
    inst.tagger.setInputDelay.assert_called_once_with(1, -12500)


def test_configure_for_flim_requires_tagger():
    inst = TimeTaggerInstrument()
    with pytest.raises(RuntimeError):
        inst.configure_for_flim(1, 2, 3, 4, 0.05, 0.2, 0.5, 0.6)


def test_create_flim_measurement_requires_tagger():
    inst = TimeTaggerInstrument()
    with pytest.raises(RuntimeError):
        inst.create_flim_measurement(1, 2, 3, 4, n_pixels=16, n_bins=10, binwidth_ps=100)
