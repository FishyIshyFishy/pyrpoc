from __future__ import annotations

import importlib
import sys
import types

import numpy as np

from pyrpoc.backend_utils.opto_control_contexts import MaskContext


def _import_daq_helpers():
    try:
        import nidaqmx  # noqa: F401
    except Exception:
        nidaqmx_stub = types.ModuleType("nidaqmx")
        nidaqmx_stub.Task = object
        constants_stub = types.ModuleType("nidaqmx.constants")
        constants_stub.AcquisitionType = types.SimpleNamespace(FINITE="FINITE")
        nidaqmx_stub.constants = constants_stub
        sys.modules["nidaqmx"] = nidaqmx_stub
        sys.modules["nidaqmx.constants"] = constants_stub
    return importlib.import_module("pyrpoc.modalities.acquisition_functions.daq_helpers")


def test_preprocess_mask_exact_shape_keeps_alignment():
    helpers = _import_daq_helpers()
    source = np.array(
        [
            [0, 255, 0, 255],
            [255, 0, 0, 255],
        ],
        dtype=np.uint8,
    )

    padded = helpers._preprocess_mask_to_scan_grid(
        source,
        total_x=8,
        total_y=2,
        scan_x_pixels=4,
        extra_left=2,
        extra_right=2,
    )
    assert padded.shape == (2, 8)
    assert np.array_equal(padded[:, :2], np.zeros((2, 2), dtype=bool))
    assert np.array_equal(padded[:, 2:6], source > 0)
    assert np.array_equal(padded[:, 6:], np.zeros((2, 2), dtype=bool))


def test_preprocess_mask_resizes_then_inserts_into_scan_window():
    helpers = _import_daq_helpers()
    source = np.array(
        [
            [255, 0],
            [0, 255],
        ],
        dtype=np.uint8,
    )

    padded = helpers._preprocess_mask_to_scan_grid(
        source,
        total_x=7,
        total_y=3,
        scan_x_pixels=4,
        extra_left=1,
        extra_right=2,
    )
    assert padded.shape == (3, 7)
    assert np.array_equal(padded[:, :1], np.zeros((3, 1), dtype=bool))
    assert np.array_equal(padded[:, 5:], np.zeros((3, 2), dtype=bool))
    # Ensure resize produced active pixels and mapping remains inside core scan region only.
    assert np.any(padded[:, 1:5])
    assert not np.any(padded[:, :1])
    assert not np.any(padded[:, 5:])


def test_generate_mask_ttl_split_gates_to_t0_only():
    helpers = _import_daq_helpers()
    context = MaskContext(
        optocontrol_key="mask",
        alias="m1",
        mask=np.array([[255, 0], [0, 255]], dtype=np.uint8),
        daq_port=0,
        daq_line=1,
    )

    ttl_signals = helpers.generate_mask_ttl_signals_split(
        total_x=4,
        total_y=2,
        pixel_samples=5,
        extra_left=1,
        extra_right=1,
        device_name="Dev1",
        mask_contexts=[context],
        scan_x_pixels=2,
        t0_samples=2,
        debug=False,
    )
    ttl = ttl_signals["Dev1/port0/line1"].reshape(2, 4, 5)
    assert np.all(ttl[:, :, 2:] == 0)
    assert np.any(ttl[:, 1:3, :2])


def test_extract_kept_samples_preserves_pixel_order():
    helpers = _import_daq_helpers()
    total_y = 2
    x_pixels = 4
    extra_left = 2
    extra_right = 1
    pixel_samples = 3
    total_x = x_pixels + extra_left + extra_right
    row_samples = total_x * pixel_samples

    source = np.arange(total_y * row_samples, dtype=np.float32)
    kept = helpers._extract_kept_samples(
        source,
        total_y=total_y,
        total_x=total_x,
        pixel_samples=pixel_samples,
        extra_left=extra_left,
        x_pixels=x_pixels,
    )

    expected = source.reshape(total_y, total_x, pixel_samples)[:, extra_left : extra_left + x_pixels, :].reshape(
        total_y, x_pixels * pixel_samples
    )
    assert np.array_equal(kept, expected)
