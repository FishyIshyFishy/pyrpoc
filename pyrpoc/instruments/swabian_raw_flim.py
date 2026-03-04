"""Minimal Swabian raw-FLIM acquisition helpers.

This module expects:
- channel 1: laser sync
- channel 2: detector clicks
- channel 3: a narrow TTL pulse at the start of each real image pixel

Detector clicks are assigned into a pixel by using the known pixel dwell time,
not by waiting for the next pixel pulse.
"""

from __future__ import annotations

import time

import numpy as np

LASER_CH = 1
DET_CH = 2
PIXEL_CH = 3

X_PIXELS = 512
Y_PIXELS = 512
N_FRAMES = 1

LASER_FREQ_HZ = 80e6
PIXEL_DWELL_PS = int(10e6)

STREAM_BUFFER_SIZE = 1_000_000
LASER_TRIGGER_V = 1.0
DET_TRIGGER_V = 1.0
PIXEL_TRIGGER_V = 1.0
USE_CONDITIONAL_FILTER = True

LASER_INPUT_DELAY_PS = 0
DET_INPUT_DELAY_PS = 0
PIXEL_INPUT_DELAY_PS = 0

FINAL_PIXEL_MARGIN_S = 1e-3
POLL_SLEEP_S = 1e-4

__all__ = ["acquire_one_frame", "acquire_n_frames"]


def _require_timetagger():
    try:
        from Swabian import TimeTagger
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Swabian Python bindings are not installed. Import failed at "
            "'from Swabian import TimeTagger'."
        ) from exc
    return TimeTagger


def _laser_period_ps(laser_frequency_hz):
    return int(round(1e12 / float(laser_frequency_hz)))


def _final_pixel_timeout_s(pixel_dwell_ps):
    return (pixel_dwell_ps / 1e12) + FINAL_PIXEL_MARGIN_S


def _new_pixel_lists(total_pixels):
    return [[] for _ in range(total_pixels)]


def _pixel_lists_to_frame(pixel_lists, y_pixels, x_pixels):
    frame = np.empty((y_pixels, x_pixels), dtype=object)
    for flat_index, pixel_values in enumerate(pixel_lists):
        y_index, x_index = divmod(flat_index, x_pixels)
        frame[y_index, x_index] = np.asarray(pixel_values, dtype=np.int64)
    return frame


def _configure_tagger(
    tagger,
    laser_ch,
    det_ch,
    pixel_ch,
    laser_trigger_v,
    det_trigger_v,
    pixel_trigger_v,
    use_conditional_filter,
):
    tagger.setTriggerLevel(laser_ch, laser_trigger_v)
    tagger.setTriggerLevel(det_ch, det_trigger_v)
    tagger.setTriggerLevel(pixel_ch, pixel_trigger_v)

    laser_delay_ps = int(LASER_INPUT_DELAY_PS)
    if use_conditional_filter:
        # Conditional filtering on the detector typically passes the next laser
        # pulse. Shift it back by one laser period so detector - laser gives a
        # normal FLIM delay.
        laser_delay_ps -= _laser_period_ps(LASER_FREQ_HZ)
        tagger.setConditionalFilter(trigger=[det_ch], filtered=[laser_ch])

    if laser_delay_ps:
        tagger.setInputDelay(laser_ch, laser_delay_ps)
    if DET_INPUT_DELAY_PS:
        tagger.setInputDelay(det_ch, int(DET_INPUT_DELAY_PS))
    if PIXEL_INPUT_DELAY_PS:
        tagger.setInputDelay(pixel_ch, int(PIXEL_INPUT_DELAY_PS))


def _raise_on_bad_chunk(data, buffer_size):
    if data.size == buffer_size:
        raise RuntimeError(
            "TimeTagStream buffer filled completely. Increase STREAM_BUFFER_SIZE "
            "or reduce the incoming tag rate."
        )

    event_types = data.getEventTypes()
    non_timetag = event_types != 0
    if np.any(non_timetag):
        bad_types = np.unique(event_types[non_timetag]).tolist()
        missed = data.getMissedEvents()
        missed_total = int(np.sum(missed[non_timetag]))
        raise RuntimeError(
            "Received non-TimeTag events from the Time Tagger stream. "
            f"event_types={bad_types}, missed_events={missed_total}"
        )


def _process_chunk(
    channels,
    timestamps,
    laser_ch,
    det_ch,
    pixel_ch,
    total_pixels,
    pixel_dwell_ps,
    pixel_lists,
    current_pixel_index,
    pixel_count_in_frame,
    current_pixel_start_ps,
    current_pixel_end_ps,
    last_laser_time,
    frames,
    x_pixels,
    y_pixels,
    n_frames,
):
    finalize_deadline = None

    for timestamp, channel in zip(timestamps, channels):
        timestamp = int(timestamp)
        channel = int(channel)

        if channel == pixel_ch:
            if pixel_count_in_frame == total_pixels:
                frames.append(_pixel_lists_to_frame(pixel_lists, y_pixels, x_pixels))
                if len(frames) >= n_frames:
                    return (
                        pixel_lists,
                        current_pixel_index,
                        pixel_count_in_frame,
                        current_pixel_start_ps,
                        current_pixel_end_ps,
                        last_laser_time,
                        finalize_deadline,
                        True,
                    )

                pixel_lists = _new_pixel_lists(total_pixels)
                current_pixel_index = -1
                pixel_count_in_frame = 0
                current_pixel_start_ps = None
                current_pixel_end_ps = None
                last_laser_time = None

            current_pixel_index += 1
            pixel_count_in_frame += 1
            current_pixel_start_ps = timestamp
            current_pixel_end_ps = timestamp + int(pixel_dwell_ps)

            if pixel_count_in_frame == total_pixels:
                finalize_deadline = time.monotonic() + _final_pixel_timeout_s(pixel_dwell_ps)

        elif channel == laser_ch:
            last_laser_time = timestamp

        elif channel == det_ch:
            if current_pixel_index < 0 or current_pixel_index >= total_pixels:
                continue
            if last_laser_time is None:
                continue
            if current_pixel_start_ps is None or current_pixel_end_ps is None:
                continue
            if timestamp < current_pixel_start_ps or timestamp >= current_pixel_end_ps:
                continue
            if timestamp < last_laser_time:
                continue

            pixel_lists[current_pixel_index].append(timestamp - last_laser_time)

    return (
        pixel_lists,
        current_pixel_index,
        pixel_count_in_frame,
        current_pixel_start_ps,
        current_pixel_end_ps,
        last_laser_time,
        finalize_deadline,
        False,
    )


def _acquire_frames(
    x_pixels,
    y_pixels,
    n_frames,
    pixel_dwell_ps,
    laser_ch,
    det_ch,
    pixel_ch,
    laser_trigger_v,
    det_trigger_v,
    pixel_trigger_v,
    use_conditional_filter,
):
    if x_pixels <= 0 or y_pixels <= 0:
        raise ValueError("x_pixels and y_pixels must both be positive.")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive.")
    if pixel_dwell_ps <= 0:
        raise ValueError("pixel_dwell_ps must be positive.")

    TimeTagger = _require_timetagger()
    tagger = TimeTagger.createTimeTagger()
    stream = None
    frames = []

    total_pixels = int(x_pixels) * int(y_pixels)
    pixel_lists = _new_pixel_lists(total_pixels)
    current_pixel_index = -1
    pixel_count_in_frame = 0
    current_pixel_start_ps = None
    current_pixel_end_ps = None
    last_laser_time = None
    finalize_deadline = None

    try:
        _configure_tagger(
            tagger,
            laser_ch=laser_ch,
            det_ch=det_ch,
            pixel_ch=pixel_ch,
            laser_trigger_v=laser_trigger_v,
            det_trigger_v=det_trigger_v,
            pixel_trigger_v=pixel_trigger_v,
            use_conditional_filter=use_conditional_filter,
        )

        stream = TimeTagger.TimeTagStream(
            tagger=tagger,
            n_max_events=STREAM_BUFFER_SIZE,
            channels=[laser_ch, det_ch, pixel_ch],
        )
        stream.start()

        while len(frames) < n_frames:
            data = stream.getData()
            if data.size > 0:
                _raise_on_bad_chunk(data, STREAM_BUFFER_SIZE)

                (
                    pixel_lists,
                    current_pixel_index,
                    pixel_count_in_frame,
                    current_pixel_start_ps,
                    current_pixel_end_ps,
                    last_laser_time,
                    new_finalize_deadline,
                    done,
                ) = _process_chunk(
                    channels=data.getChannels(),
                    timestamps=data.getTimestamps(),
                    laser_ch=laser_ch,
                    det_ch=det_ch,
                    pixel_ch=pixel_ch,
                    total_pixels=total_pixels,
                    pixel_dwell_ps=pixel_dwell_ps,
                    pixel_lists=pixel_lists,
                    current_pixel_index=current_pixel_index,
                    pixel_count_in_frame=pixel_count_in_frame,
                    current_pixel_start_ps=current_pixel_start_ps,
                    current_pixel_end_ps=current_pixel_end_ps,
                    last_laser_time=last_laser_time,
                    frames=frames,
                    x_pixels=x_pixels,
                    y_pixels=y_pixels,
                    n_frames=n_frames,
                )

                if new_finalize_deadline is not None:
                    finalize_deadline = new_finalize_deadline

                if done:
                    break

            if pixel_count_in_frame == total_pixels and finalize_deadline is not None:
                if time.monotonic() >= finalize_deadline:
                    frames.append(_pixel_lists_to_frame(pixel_lists, y_pixels, x_pixels))
                    if len(frames) >= n_frames:
                        break

                    pixel_lists = _new_pixel_lists(total_pixels)
                    current_pixel_index = -1
                    pixel_count_in_frame = 0
                    current_pixel_start_ps = None
                    current_pixel_end_ps = None
                    last_laser_time = None
                    finalize_deadline = None
                    continue

            if data.size == 0:
                time.sleep(POLL_SLEEP_S)

        return frames
    finally:
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
        TimeTagger.freeTimeTagger(tagger)


def acquire_one_frame(
    x_pixels,
    y_pixels,
    pixel_dwell_ps=PIXEL_DWELL_PS,
    laser_ch=LASER_CH,
    det_ch=DET_CH,
    pixel_ch=PIXEL_CH,
    laser_trigger_v=LASER_TRIGGER_V,
    det_trigger_v=DET_TRIGGER_V,
    pixel_trigger_v=PIXEL_TRIGGER_V,
    use_conditional_filter=USE_CONDITIONAL_FILTER,
):
    """Acquire one frame of per-pixel FLIM delays.

    Returns a `(y_pixels, x_pixels)` object array where each element is a 1D
    `np.int64` array of laser-relative photon delays in ps.
    """

    return _acquire_frames(
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        n_frames=1,
        pixel_dwell_ps=pixel_dwell_ps,
        laser_ch=laser_ch,
        det_ch=det_ch,
        pixel_ch=pixel_ch,
        laser_trigger_v=laser_trigger_v,
        det_trigger_v=det_trigger_v,
        pixel_trigger_v=pixel_trigger_v,
        use_conditional_filter=use_conditional_filter,
    )[0]


def acquire_n_frames(
    x_pixels,
    y_pixels,
    n_frames=N_FRAMES,
    pixel_dwell_ps=PIXEL_DWELL_PS,
    laser_ch=LASER_CH,
    det_ch=DET_CH,
    pixel_ch=PIXEL_CH,
    laser_trigger_v=LASER_TRIGGER_V,
    det_trigger_v=DET_TRIGGER_V,
    pixel_trigger_v=PIXEL_TRIGGER_V,
    use_conditional_filter=USE_CONDITIONAL_FILTER,
):
    """Acquire multiple frames of per-pixel FLIM delays."""

    return _acquire_frames(
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        n_frames=n_frames,
        pixel_dwell_ps=pixel_dwell_ps,
        laser_ch=laser_ch,
        det_ch=det_ch,
        pixel_ch=pixel_ch,
        laser_trigger_v=laser_trigger_v,
        det_trigger_v=det_trigger_v,
        pixel_trigger_v=pixel_trigger_v,
        use_conditional_filter=use_conditional_filter,
    )
