"""Generate the phase 0 golden reference arrays.

Records what the v3.0.2 hardware arithmetic computes, so the functions moved
into ``operations/`` during phase 1 (and the simulated devices written in
phase 2) can be checked against it. See ``docs/plans/260829-implementation_plan.md``
-- phases 1 through 4 each say "identical to the phase 0 reference".

Only pure functions are captured: numbers in, numbers out, no hardware. Run
with ``python -m tests.reference.generate_references`` from the repo root, and
only ever re-run it deliberately -- regenerating on a changed implementation
would silently rebase the thing the tests compare against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyrpoc.backend_utils.opto_control_contexts import MaskContext
from pyrpoc.modalities.helpers.daq import generate_raster_waveform
from pyrpoc.modalities.confocal import acquisition_core as confocal_core
from pyrpoc.modalities.split_confocal import acquisition_core as split_core
from pyrpoc.modalities.flim import acquisition_core as flim_core

reference_path = Path(__file__).parent / "phase0_references.npz"

# One small scan geometry, used everywhere so the arrays stay inspectable.
scan = dict(x_pixels=8, y_pixels=6, extra_left=3, extra_right=2)
pixel_samples = 4
total_x = scan["x_pixels"] + scan["extra_left"] + scan["extra_right"]
total_y = scan["y_pixels"]
t0_samples, t1_samples = 1, 1


def build_references() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    # --- waveform generation (shared by all three modalities) ---
    out["raster_waveform"] = generate_raster_waveform(
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

    kept = confocal_core.extract_kept_samples(
        raw_channel, total_y, total_x, pixel_samples, scan["extra_left"], scan["x_pixels"]
    )
    out["confocal_extract_kept_samples"] = kept
    out["confocal_reshape_to_frame"] = confocal_core.reshape_to_frame(
        kept[np.newaxis], total_y, scan["x_pixels"], pixel_samples
    )

    split_kept = split_core.extract_kept_samples(
        raw_channel, total_y, total_x, pixel_samples, scan["extra_left"], scan["x_pixels"]
    )
    out["split_extract_kept_samples"] = split_kept
    split_frame, split_raw = split_core.reshape_to_split_frame(
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
    out["resize_mask_nearest"] = confocal_core.resize_mask_nearest(
        odd_mask > 0, target_h=total_y, target_w=scan["x_pixels"]
    )

    out["preprocess_mask_to_scan_grid"] = confocal_core.preprocess_mask_to_scan_grid(
        mask,
        total_x=total_x,
        total_y=total_y,
        scan_x_pixels=scan["x_pixels"],
        extra_left=scan["extra_left"],
        extra_right=scan["extra_right"],
    )

    ctx = MaskContext(optocontrol_key="mask", alias="m", mask=mask, daq_port=0, daq_line=3)
    ttl_kwargs = dict(
        total_x=total_x,
        total_y=total_y,
        pixel_samples=pixel_samples,
        extra_left=scan["extra_left"],
        extra_right=scan["extra_right"],
        device_name="Dev1",
        mask_contexts=[ctx],
        scan_x_pixels=scan["x_pixels"],
    )
    confocal_ttl = confocal_core.generate_mask_ttl_signals(**ttl_kwargs)
    split_ttl = split_core.generate_mask_ttl_signals(**ttl_kwargs, t0_samples=t0_samples)
    channel = "Dev1/port0/line3"
    out["confocal_mask_ttl"] = confocal_ttl[channel]
    out["split_mask_ttl"] = split_ttl[channel]

    # --- FLIM histogram reshaping ---
    n_bins = 5
    histograms = np.arange(total_y * total_x * n_bins, dtype=np.float32)
    out["flim_histograms_input"] = histograms
    cube = flim_core.reshape_flim_frame(
        histograms, n_bins, total_y, total_x, scan["extra_left"], scan["x_pixels"]
    )
    out["flim_reshape_frame"] = cube
    out["flim_intensity"] = flim_core.flim_intensity(cube)

    return out


def main() -> None:
    refs = build_references()
    np.savez_compressed(reference_path, **refs)
    print(f"wrote {reference_path} ({len(refs)} arrays)")
    for name, arr in sorted(refs.items()):
        print(f"  {name:34s} {str(arr.shape):16s} {arr.dtype}")


if __name__ == "__main__":
    main()
