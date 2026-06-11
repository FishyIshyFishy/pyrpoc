from __future__ import annotations

import pyrpoc.modalities  # noqa: F401  -- registers confocal/split_confocal/flim
import pytest

from pyrpoc.domain.app_state import AppState
from pyrpoc.services.instrument_service import InstrumentService
from pyrpoc.services.modality_service import ModalityService


def make_service(qapp) -> ModalityService:
    app_state = AppState()
    instruments = InstrumentService(app_state)
    return ModalityService(instruments, app_state)


def test_params_remembered_per_modality_across_switching(qapp):
    svc = make_service(qapp)

    svc.select_modality("confocal")
    svc.set_parameter_values({"X Pixels": 128, "Y Pixels": 64})

    svc.select_modality("split_confocal")
    assert svc.get_parameter_values() == {}  # a different modality starts clean
    svc.set_parameter_values({"X Pixels": 32})

    # Switching back restores the first modality's values...
    svc.select_modality("confocal")
    assert svc.get_parameter_values() == {"X Pixels": 128, "Y Pixels": 64}
    # ...and the second modality independently keeps its own.
    svc.select_modality("split_confocal")
    assert svc.get_parameter_values() == {"X Pixels": 32}


def test_configured_params_property_reads_selected_modality(qapp):
    svc = make_service(qapp)
    svc.select_modality("confocal")
    svc.set_parameter_values({"X Pixels": 200})
    state = svc.app_state.modality
    assert [pv.label for pv in state.configured_params] == ["X Pixels"]
    assert state.params_by_modality["confocal"][0].value == 200


def test_unknown_modality_clears_selection_but_keeps_map(qapp):
    svc = make_service(qapp)
    svc.select_modality("confocal")
    svc.set_parameter_values({"X Pixels": 10})

    with pytest.raises(KeyError):
        svc.select_modality("does_not_exist")

    assert svc.app_state.modality.selected_key is None
    assert svc.get_parameter_values() == {}
    # The remembered confocal params survive for when it is reselected.
    assert svc.app_state.modality.params_by_modality["confocal"][0].value == 10
