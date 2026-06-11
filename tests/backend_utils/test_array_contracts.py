from __future__ import annotations

import numpy as np

from pyrpoc.backend_utils.array_contracts import (
    contract_chw_float32,
    contract_hw_float32,
    infer_array_contract,
    matches_array_contract,
)


def test_infer_chw_float32():
    data = np.zeros((3, 4, 5), dtype=np.float32)
    assert infer_array_contract(data) == contract_chw_float32


def test_infer_hw_float32():
    data = np.zeros((4, 5), dtype=np.float32)
    assert infer_array_contract(data) == contract_hw_float32


def test_infer_rejects_wrong_dtype():
    assert infer_array_contract(np.zeros((4, 5), dtype=np.float64)) is None


def test_infer_rejects_non_array():
    assert infer_array_contract([1, 2, 3]) is None
    assert infer_array_contract(None) is None


def test_infer_rejects_unsupported_ndim():
    assert infer_array_contract(np.zeros((2, 3, 4, 5), dtype=np.float32)) is None
    assert infer_array_contract(np.zeros((5,), dtype=np.float32)) is None


def test_matches_chw_contract():
    assert matches_array_contract(np.zeros((3, 4, 5), dtype=np.float32), contract_chw_float32)


def test_matches_rejects_zero_sized_dims():
    assert not matches_array_contract(np.zeros((0, 4, 5), dtype=np.float32), contract_chw_float32)
    assert not matches_array_contract(np.zeros((4, 0), dtype=np.float32), contract_hw_float32)


def test_matches_rejects_wrong_contract_for_shape():
    hw = np.zeros((4, 5), dtype=np.float32)
    assert not matches_array_contract(hw, contract_chw_float32)


def test_matches_unknown_contract_is_false():
    assert not matches_array_contract(np.zeros((4, 5), dtype=np.float32), "nonsense")


def test_matches_non_array_is_false():
    assert not matches_array_contract([1, 2], contract_hw_float32)
