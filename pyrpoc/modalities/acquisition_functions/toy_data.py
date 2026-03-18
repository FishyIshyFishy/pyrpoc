from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pyrpoc.backend_utils.opto_control_contexts import MaskContext


def generate_toy_confocal_frame(
    *,
    x_pixels: int,
    y_pixels: int,
    active_channels: list[int],
    frame_index: int,
    mask_contexts: list[MaskContext],
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
) -> np.ndarray:
    if x_pixels <= 0 or y_pixels <= 0:
        raise ValueError("Scan dimensions must be positive")

    if not active_channels:
        raise ValueError("No active AI channels configured")

    frame = np.zeros((len(active_channels), y_pixels, x_pixels), dtype=np.float32)
    fast_amp = max(float(fast_axis_amplitude), 1e-6)
    slow_amp = max(float(slow_axis_amplitude), 1e-6)

    for channel_index, ai_channel in enumerate(active_channels):
        channel = _build_toy_channel(
            x_pixels=x_pixels,
            y_pixels=y_pixels,
            frame_index=frame_index,
            channel_index=channel_index,
            ai_channel=ai_channel,
            fast_axis_offset=float(fast_axis_offset),
            fast_axis_amplitude=fast_amp,
            slow_axis_offset=float(slow_axis_offset),
            slow_axis_amplitude=slow_amp,
        )
        frame[channel_index] = channel

    _apply_masks(frame, mask_contexts)
    return frame


def generate_toy_split_confocal_frame(
    *,
    x_pixels: int,
    y_pixels: int,
    active_channels: list[int],
    frame_index: int,
    mask_contexts: list[MaskContext],
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
    t0_samples: int,
    t1_samples: int,
    pixel_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if pixel_samples <= 0:
        raise ValueError("pixel_samples must be positive")
    if t0_samples < 1:
        raise ValueError("t0 samples must be >= 1")
    if t1_samples < 0:
        raise ValueError("t1 samples must be >= 0")
    if t0_samples + t1_samples >= pixel_samples:
        raise ValueError(
            f"split timing uses all samples: t0={t0_samples}, t1={t1_samples}, pixel_samples={pixel_samples}"
        )

    frame = generate_toy_confocal_frame(
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        active_channels=active_channels,
        frame_index=frame_index,
        mask_contexts=mask_contexts,
        fast_axis_offset=fast_axis_offset,
        fast_axis_amplitude=fast_axis_amplitude,
        slow_axis_offset=slow_axis_offset,
        slow_axis_amplitude=slow_axis_amplitude,
    )

    rng = np.random.default_rng(seed=(frame_index + 1) * 17)
    raw = np.repeat(frame[:, :, :, None], pixel_samples, axis=3)
    raw = raw + rng.normal(0.0, 0.02, size=raw.shape).astype(np.float32)

    first_start = int(t0_samples)
    second_start = first_start + int(t1_samples)

    first_portion = raw[:, :, :, :first_start].mean(axis=3)
    if second_start < pixel_samples:
        second_portion = raw[:, :, :, second_start:].mean(axis=3)
    else:
        second_portion = np.zeros_like(first_portion)

    first_portion = _overlay_split_label(first_portion, "1")
    second_portion = _overlay_split_label(second_portion, "2")

    split_frame = np.stack((first_portion, second_portion), axis=1).reshape(frame.shape[0] * 2, y_pixels, x_pixels)
    return split_frame.astype(np.float32, copy=False), raw.astype(np.float32, copy=False)


def _build_toy_channel(
    *,
    x_pixels: int,
    y_pixels: int,
    frame_index: int,
    channel_index: int,
    ai_channel: int,
    fast_axis_offset: float,
    fast_axis_amplitude: float,
    slow_axis_offset: float,
    slow_axis_amplitude: float,
) -> np.ndarray:
    seed = (frame_index + 1) * 1009 + (channel_index + 1) * 101 + ai_channel * 17
    rng = np.random.default_rng(seed)
    channel = np.zeros((y_pixels, x_pixels), dtype=np.float32)

    x = np.linspace(-1.0, 1.0, x_pixels, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, y_pixels, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    x_term = (xx - fast_axis_offset) / fast_axis_amplitude
    y_term = (yy - slow_axis_offset) / slow_axis_amplitude
    channel += 0.15 * np.sin((x_term + 0.07 * frame_index) * (5.0 + channel_index))
    channel += 0.12 * np.cos((y_term - 0.05 * frame_index) * (4.0 + 0.5 * channel_index))

    n_shapes = int(rng.integers(10, 18))
    for _ in range(n_shapes):
        intensity = float(rng.uniform(0.2, 1.0))
        if rng.random() < 0.5:
            cx = int(rng.integers(0, x_pixels))
            cy = int(rng.integers(0, y_pixels))
            radius = int(rng.integers(max(3, min(x_pixels, y_pixels) // 30), max(6, min(x_pixels, y_pixels) // 8)))
            cv2.circle(
                channel,
                (cx, cy),
                radius,
                intensity,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        else:
            cx = int(rng.integers(0, x_pixels))
            cy = int(rng.integers(0, y_pixels))
            a = int(rng.integers(max(4, x_pixels // 40), max(8, x_pixels // 10)))
            b = int(rng.integers(max(4, y_pixels // 40), max(8, y_pixels // 10)))
            angle = float(rng.uniform(0.0, 180.0))
            cv2.ellipse(
                channel,
                (cx, cy),
                (a, b),
                angle,
                0.0,
                360.0,
                intensity,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    channel += rng.normal(0.0, 0.03, size=(y_pixels, x_pixels)).astype(np.float32)
    channel -= float(np.min(channel))
    peak = float(np.max(channel))
    if peak > 0:
        channel /= peak

    return channel.astype(np.float32, copy=False)


def _apply_masks(
    frame: np.ndarray,
    mask_contexts: list[MaskContext],
) -> None:
    if frame.ndim != 3:
        return
    _, h, w = frame.shape

    for context in mask_contexts:
        mask = np.asarray(context.mask, dtype=np.uint8) #pyright:ignore
        if mask.ndim != 2:
            continue

        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        active = mask > 0
        if not np.any(active):
            continue

        for idx in range(frame.shape[0]):
            boost = float(np.max(frame[idx]))
            frame[idx, active] += boost


def _overlay_split_label(image: np.ndarray, label: str) -> np.ndarray:
    if image.ndim != 2:
        return image

    if image.shape[0] < 16 or image.shape[1] < 24:
        return image

    marker = np.zeros(image.shape[:2], dtype=np.uint8)
    baseline = 0.8 * image.shape[0] if image.shape[0] > 0 else 0
    x = max(2, int(image.shape[1] * 0.05))
    y = int(max(14, baseline))
    cv2.putText(
        marker,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        255,
        2,
        cv2.LINE_AA,
    )

    if not np.any(marker):
        return image

    weight = 0.2
    return image + weight * (marker.astype(np.float32) / 255.0) * (float(np.max(np.abs(image))) + 1.0)
