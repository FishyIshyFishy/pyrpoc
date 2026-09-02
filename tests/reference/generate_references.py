"""Generate the phase 0 golden reference arrays.

Records what the v3.0.2 hardware arithmetic computes, so the functions moved
into ``programs/hardware/`` can be checked against it. See
``docs/plans/260829-implementation_plan.md``.

Only pure functions are captured: numbers in, numbers out, no hardware.

**Do not re-run this.** ``phase0_references.npz`` was written from the v3.0.2
implementation and is the thing the tests compare against; regenerating it on a
changed implementation silently rebases the comparison. This module is imported
by ``test_phase0_references.py`` to compute the *current* values only.

The imports point at ``programs/hardware/``. They pointed at ``operations/``
before that folder was demoted into ``programs/``, and at the v3.0 modalities
before that. The arrays must be unchanged across every one of those moves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyrpoc.core.modulation import MaskBinding
from pyrpoc.core.params import ScanGroup
from pyrpoc.programs.hardware import modulation as ops_modulation
from pyrpoc.programs.hardware import raster as ops_raster
from pyrpoc.programs.hardware import split_raster as ops_split
from pyrpoc.programs.hardware import tagger as ops_tagger

reference_path = Path(__file__).parent / "phase0_references.npz"

# One small scan geometry, used everywhere so the arrays stay inspectable.
scan = dict(x_pixels=8, y_pixels=6, extra_left=3, extra_right=2)
pixel_samples = 4
total_x = scan["x_pixels"] + scan["extra_left"] + scan["extra_right"]
total_y = scan["y_pixels"]
t0_samples, t1_samples = 1, 1


def build_references() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    # --- waveform generation (shared by all three programs) ---
    out["raster_waveform"] = ops_raster.generate_raster_waveform(
        pixel_samples=pixel_samples,
        fast_axis_offset=0.25,
        fast_axis_amplitude=1.5,
        slow_axis_offset=-0.5,
        slow_axis_amplitude=2.0,
        **scan,
    )

    # --- deterministic stand-in for one AI channel's raw sample stream ---
    n_samples = total_y * total_x * pixel_samples
    raw_channel = np.arange(n_samples, dtype=np.float32).reshape(total_y, total_x * pixel_samples)
    out["raw_channel_input"] = raw_channel

    kept = ops_raster.extract_kept_samples(
        raw_channel, total_y, total_x, pixel_samples, scan["extra_left"], scan["x_pixels"]
    )
    out["confocal_extract_kept_samples"] = kept
    out["confocal_reshape_to_frame"] = ops_raster.reshape_to_frame(
        kept[np.newaxis], total_y, scan["x_pixels"], pixel_samples
    )

    # The split copy of extract_kept_samples was duplicated verbatim; phase 1
    # collapsed the two into one, so both names now record the same function.
    split_kept = ops_raster.extract_kept_samples(
        raw_channel, total_y, total_x, pixel_samples, scan["extra_left"], scan["x_pixels"]
    )
    out["split_extract_kept_samples"] = split_kept
    split_frame, split_raw = ops_split.reshape_to_split_frame(
        split_kept[np.newaxis], total_y, scan["x_pixels"], pixel_samples, t0_samples, t1_samples
    )
    out["split_reshape_frame"] = split_frame
    out["split_reshape_raw"] = split_raw

    # --- mask -> TTL path (the part least covered by the automatic tests) ---
    mask = np.zeros((total_y, scan["x_pixels"]), dtype=np.uint8)
    mask[1:4, 2:6] = 255
    out["mask_input"] = mask

    odd_mask = np.zeros((5, 7), dtype=np.uint8)
    odd_mask[1:4, 2:5] = 1
    out["mask_input_odd"] = odd_mask
    out["resize_mask_nearest"] = ops_modulation.resize_mask_nearest(
        odd_mask > 0, target_h=total_y, target_w=scan["x_pixels"]
    )

    out["preprocess_mask_to_scan_grid"] = ops_modulation.preprocess_mask_to_scan_grid(
        mask,
        total_x=total_x,
        total_y=total_y,
        scan_x_pixels=scan["x_pixels"],
        extra_left=scan["extra_left"],
        extra_right=scan["extra_right"],
    )

    scan_group = ScanGroup(
        x_pixels=scan["x_pixels"],
        y_pixels=scan["y_pixels"],
        extra_left=scan["extra_left"],
        extra_right=scan["extra_right"],
    )
    bindings = [(MaskBinding(path=Path("mask.png"), port=0, line=3), mask)]
    ttl_kwargs = dict(scan=scan_group, pixel_samples=pixel_samples, device_name="Dev1")
    confocal_ttl = ops_modulation.mask_ttl(bindings, **ttl_kwargs)
    split_ttl = ops_modulation.split_mask_ttl(bindings, **ttl_kwargs, t0_samples=t0_samples)
    channel = "Dev1/port0/line3"
    out["confocal_mask_ttl"] = confocal_ttl[channel]
    out["split_mask_ttl"] = split_ttl[channel]

    # --- FLIM histogram reshaping ---
    n_bins = 5
    histograms = np.arange(total_y * total_x * n_bins, dtype=np.float32)
    out["flim_histograms_input"] = histograms
    cube = ops_tagger.reshape_flim_frame(
        histograms, n_bins, total_y, total_x, scan["extra_left"], scan["x_pixels"]
    )
    out["flim_reshape_frame"] = cube
    out["flim_intensity"] = ops_tagger.flim_intensity(cube)

    return out


def main() -> None:
    refs = build_references()
    np.savez_compressed(reference_path, **refs)
    print(f"wrote {reference_path} ({len(refs)} arrays)")
    for name, arr in sorted(refs.items()):
        print(f"  {name:34s} {str(arr.shape):16s} {arr.dtype}")


if __name__ == "__main__":
    main()
