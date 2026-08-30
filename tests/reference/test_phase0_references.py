"""Pin the hardware arithmetic to the phase 0 golden arrays.

Every function here is moved into ``operations/`` during phase 1. These tests
are the check that the move did not change what is computed -- point them at
the new functions once they exist and they must still pass unchanged.

If one of these fails, the arithmetic that reaches the instrument changed.
Fix the implementation, do not regenerate the references.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.reference.generate_references import build_references, reference_path


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    if not reference_path.exists():
        pytest.fail(
            f"missing {reference_path}; regenerate with "
            "`python -m tests.reference.generate_references`"
        )
    with np.load(reference_path) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def current() -> dict[str, np.ndarray]:
    return build_references()


reference_names = [
    "raster_waveform",
    "confocal_extract_kept_samples",
    "confocal_reshape_to_frame",
    "split_extract_kept_samples",
    "split_reshape_frame",
    "split_reshape_raw",
    "resize_mask_nearest",
    "preprocess_mask_to_scan_grid",
    "confocal_mask_ttl",
    "split_mask_ttl",
    "flim_reshape_frame",
    "flim_intensity",
]


@pytest.mark.parametrize("name", reference_names)
def test_matches_phase0_reference(golden, current, name):
    assert name in golden, f"{name} missing from the stored references"
    expected, actual = golden[name], current[name]
    assert actual.shape == expected.shape, f"{name}: shape drifted"
    assert actual.dtype == expected.dtype, f"{name}: dtype drifted"
    np.testing.assert_array_equal(actual, expected, err_msg=f"{name} changed")


def test_reference_file_covers_every_checked_name(golden):
    assert set(reference_names) <= set(golden), "reference file is missing entries"


# --- properties the arithmetic must hold, independent of the stored arrays ---

def test_confocal_and_split_share_identical_sample_extraction(current):
    """The two extract_kept_samples copies are duplicated verbatim today;
    phase 1 collapses them into one operation."""
    np.testing.assert_array_equal(
        current["confocal_extract_kept_samples"],
        current["split_extract_kept_samples"],
    )


def test_split_ttl_is_confocal_ttl_gated_to_t0(current):
    """Split confocal's only TTL difference is truncation to the first
    t0_samples of each pixel, so its TTL must be a subset of confocal's."""
    confocal_ttl = current["confocal_mask_ttl"]
    split_ttl = current["split_mask_ttl"]
    assert not np.any(split_ttl & ~confocal_ttl), "split TTL fires outside the confocal TTL"
    assert split_ttl.sum() < confocal_ttl.sum(), "split TTL was not gated down"


def test_flim_intensity_is_the_histogram_sum(current):
    np.testing.assert_array_equal(
        current["flim_intensity"], current["flim_reshape_frame"].sum(axis=2)
    )
