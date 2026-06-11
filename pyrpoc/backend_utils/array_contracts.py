from __future__ import annotations

import numpy as np


contract_chw_float32 = "chw_float32"
contract_hw_float32 = "hw_float32"


def infer_array_contract(data: object) -> str | None:
    if not isinstance(data, np.ndarray):
        return None
    if data.ndim == 3 and data.dtype == np.float32:
        return contract_chw_float32
    if data.ndim == 2 and data.dtype == np.float32:
        return contract_hw_float32
    return None


def matches_array_contract(data: object, contract: str) -> bool:
    if not isinstance(data, np.ndarray):
        return False

    if contract == contract_chw_float32:
        return (
            data.ndim == 3
            and data.dtype == np.float32
            and data.shape[0] > 0
            and data.shape[1] > 0
            and data.shape[2] > 0
        )
    if contract == contract_hw_float32:
        return data.ndim == 2 and data.dtype == np.float32 and data.shape[0] > 0 and data.shape[1] > 0
    return False
