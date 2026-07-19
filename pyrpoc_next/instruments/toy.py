"""Synthetic data generators for the simulated instruments. Simulation only.

Kept isolated here so no real-hardware path ever imports simulation code.
"""

from __future__ import annotations

import cv2
import numpy as np


def toy_channel(x_pixels: int, y_pixels: int, frame_index: int, channel_index: int) -> np.ndarray:
    """A deterministic synthetic image channel: gradients plus random blobs, in [0, 1]."""
    seed = (frame_index + 1) * 1009 + (channel_index + 1) * 101
    rng = np.random.default_rng(seed)
    channel = np.zeros((y_pixels, x_pixels), dtype=np.float32)

    x = np.linspace(-1.0, 1.0, x_pixels, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, y_pixels, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    channel += 0.15 * np.sin((xx + 0.07 * frame_index) * (5.0 + channel_index))
    channel += 0.12 * np.cos((yy - 0.05 * frame_index) * (4.0 + 0.5 * channel_index))

    for _ in range(int(rng.integers(10, 18))):
        intensity = float(rng.uniform(0.2, 1.0))
        cx, cy = int(rng.integers(0, x_pixels)), int(rng.integers(0, y_pixels))
        radius = int(rng.integers(max(3, min(x_pixels, y_pixels) // 30), max(6, min(x_pixels, y_pixels) // 8)))
        cv2.circle(channel, (cx, cy), radius, intensity, thickness=-1, lineType=cv2.LINE_AA)

    channel += rng.normal(0.0, 0.03, size=(y_pixels, x_pixels)).astype(np.float32)
    channel -= float(channel.min())
    peak = float(channel.max())
    if peak > 0:
        channel /= peak
    return channel.astype(np.float32, copy=False)


def toy_frame(x_pixels: int, y_pixels: int, channel_count: int, frame_index: int) -> np.ndarray:
    """A synthetic (channels, height, width) intensity frame."""
    channels = [toy_channel(x_pixels, y_pixels, frame_index, index) for index in range(channel_count)]
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def toy_flim_cube(
    x_pixels: int, y_pixels: int, n_bins: int, bin_width_ps: float, laser_period_ps: float, frame_index: int
) -> np.ndarray:
    """A synthetic (height, width, bins) decay-histogram cube; brighter pixels live longer."""
    intensity = toy_channel(x_pixels, y_pixels, frame_index, 0)
    bin_centers = (np.arange(n_bins, dtype=np.float32) + 0.5) * float(bin_width_ps)
    peak_ps = 0.05 * float(laser_period_ps)
    tau_map = 800.0 + 2200.0 * intensity
    decay = np.exp(-np.clip(bin_centers - peak_ps, 0.0, None)[None, None, :] / tau_map[:, :, None])
    decay[:, :, bin_centers < peak_ps] = 0.0
    counts = decay * (50.0 * intensity[:, :, None] + 1.0)
    return counts.astype(np.float32, copy=False)


def boost_masked_pixels(frame: np.ndarray, mask: np.ndarray) -> None:
    """Brighten pixels where a mask is active, in place — the toy analogue of a mask TTL."""
    channels, height, width = frame.shape
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    active = mask > 0
    for index in range(channels):
        frame[index, active] += float(frame[index].max())
