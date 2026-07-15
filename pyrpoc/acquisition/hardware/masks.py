from __future__ import annotations

import numpy as np

from pyrpoc.structs.contexts import MaskContext


def resize_mask_nearest(mask_bool: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    source_h, source_w = mask_bool.shape
    if source_h <= 0 or source_w <= 0:
        return np.zeros((target_h, target_w), dtype=bool)
    y_idx = np.minimum((np.arange(target_h, dtype=np.int64) * source_h) // target_h, source_h - 1)
    x_idx = np.minimum((np.arange(target_w, dtype=np.int64) * source_w) // target_w, source_w - 1)
    return mask_bool[np.ix_(y_idx, x_idx)]


def preprocess_mask_to_scan_grid(
    raw_mask: object,
    total_x: int,
    total_y: int,
    scan_x_pixels: int,
    extra_left: int,
    extra_right: int,
) -> np.ndarray:
    if total_x <= 0 or total_y <= 0:
        raise ValueError("total_x and total_y must be positive")
    if (extra_left + scan_x_pixels + extra_right) != total_x:
        raise ValueError("total_x must equal extra_left + scan_x_pixels + extra_right")

    mask = np.asarray(raw_mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={mask.shape}")

    if scan_x_pixels == 0:
        return np.zeros((total_y, total_x), dtype=bool)

    mask_bool = mask > 0
    if mask_bool.shape != (total_y, scan_x_pixels):
        mask_bool = resize_mask_nearest(mask_bool, target_h=total_y, target_w=scan_x_pixels)

    padded = np.zeros((total_y, total_x), dtype=bool)
    padded[:, extra_left : extra_left + scan_x_pixels] = mask_bool
    return padded


def generate_mask_ttl_signals(
    *,
    total_x: int,
    total_y: int,
    pixel_samples: int,
    extra_left: int,
    extra_right: int,
    device_name: str,
    mask_contexts: list[MaskContext],
    scan_x_pixels: int,
    t0_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Turn drawn masks into per-DO-channel boolean TTL streams.

    When ``t0_samples`` is given, each pixel's TTL is truncated to its first
    ``t0_samples`` samples (used by the split-confocal gated path); otherwise
    the TTL is held for the whole pixel dwell (confocal / FLIM).
    """
    ttl_signals: dict[str, np.ndarray] = {}
    for context in mask_contexts:
        if context.mask is None:
            continue
        channel_name = f"{device_name}/port{int(context.daq_port)}/line{int(context.daq_line)}"
        try:
            padded = preprocess_mask_to_scan_grid(
                context.mask,
                total_x=total_x,
                total_y=total_y,
                scan_x_pixels=scan_x_pixels,
                extra_left=extra_left,
                extra_right=extra_right,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to preprocess mask for {channel_name}: {exc}") from exc
        if not np.any(padded):
            continue
        ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)
        ttl[padded] = True
        if t0_samples is not None and t0_samples < pixel_samples:
            ttl[:, :, t0_samples:] = False
        ttl_signals[channel_name] = ttl.reshape(-1)
    return ttl_signals
