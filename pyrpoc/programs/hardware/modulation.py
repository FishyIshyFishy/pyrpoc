"""Masks to per-pixel TTL waveforms.

Moved from the two identical copies in ``modalities/confocal`` and
``modalities/split_confocal``. Takes loaded mask arrays and bindings rather than
``MaskContext`` objects, so nothing here knows the optocontrol machinery ever
existed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pyrpoc.core.modulation import MaskBinding
from pyrpoc.core.params import ScanGroup


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


def mask_ttl(
    masks: Sequence[tuple[MaskBinding, np.ndarray]],
    *,
    scan: ScanGroup,
    pixel_samples: int,
    device_name: str,
) -> dict[str, np.ndarray]:
    """One flat boolean TTL signal per bound digital line.

    A mask that is entirely zero after padding is skipped, so its channel does
    not appear in the result and no DO task is created for it.
    """
    total_x = scan.total_x
    total_y = scan.y_pixels

    ttl_signals: dict[str, np.ndarray] = {}
    for binding, mask in masks:
        if mask is None:
            continue
        channel_name = binding.channel(device_name)
        try:
            padded = preprocess_mask_to_scan_grid(
                mask,
                total_x=total_x,
                total_y=total_y,
                scan_x_pixels=scan.x_pixels,
                extra_left=scan.extra_left,
                extra_right=scan.extra_right,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to preprocess mask for {channel_name}: {exc}") from exc
        if not np.any(padded):
            continue
        ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)
        ttl[padded] = True
        ttl_signals[channel_name] = ttl.reshape(-1)

    return ttl_signals


def split_mask_ttl(
    masks: Sequence[tuple[MaskBinding, np.ndarray]],
    *,
    scan: ScanGroup,
    pixel_samples: int,
    device_name: str,
    t0_samples: int,
) -> dict[str, np.ndarray]:
    """``mask_ttl`` truncated to the first ``t0_samples`` of every pixel.

    That gate is the only thing split confocal's TTL generation did differently,
    so the twenty lines above it are shared rather than copied.
    """
    signals = mask_ttl(masks, scan=scan, pixel_samples=pixel_samples, device_name=device_name)
    if t0_samples >= pixel_samples:
        return signals
    for signal in signals.values():
        signal.reshape(-1, pixel_samples)[:, t0_samples:] = False
    return signals
