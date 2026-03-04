from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile


def acquire_raw_flim(
    x_pixels,
    y_pixels,
    n_frames,
    pixel_dwell_ps,
    extra_left,
    extra_right,
    laser_frequency_hz,
    laser_ch,
    detector_ch,
    pixel_ch,
    laser_trigger_v,
    detector_trigger_v,
    pixel_trigger_v,
    laser_input_delay_ps=0,
    detector_input_delay_ps=0,
    pixel_input_delay_ps=0,
    laser_event_divider=1,
    use_conditional_filter=False,
    stream_buffer_size=4_000_000,
    poll_sleep_s=1e-4,
    final_pixel_margin_s=1e-3,
):
    from Swabian import TimeTagger

    if x_pixels <= 0 or y_pixels <= 0:
        raise ValueError("x_pixels and y_pixels must be positive.")
    if extra_left < 0 or extra_right < 0:
        raise ValueError("extra_left and extra_right must be non-negative.")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive.")
    if pixel_dwell_ps <= 0:
        raise ValueError("pixel_dwell_ps must be positive.")
    if laser_frequency_hz <= 0:
        raise ValueError("laser_frequency_hz must be positive.")
    if not 1 <= int(laser_event_divider) <= 65535:
        raise ValueError("laser_event_divider must be in the range [1, 65535].")

    total_x_pixels = int(x_pixels) + int(extra_left) + int(extra_right)
    total_pixels = total_x_pixels * int(y_pixels)
    frame_duration_ps = int(total_pixels * int(pixel_dwell_ps))

    def new_pixel_lists():
        return [[] for _ in range(total_pixels)]

    def finalize_frame(pixel_lists):
        frame = np.empty((y_pixels, total_x_pixels), dtype=object)
        for flat_index, delays in enumerate(pixel_lists):
            y_index, x_index = divmod(flat_index, total_x_pixels)
            frame[y_index, x_index] = np.asarray(delays, dtype=np.int64)
        return frame[:, extra_left:extra_left + x_pixels]

    tagger = TimeTagger.createTimeTagger()
    stream = None
    frames = []
    stats = {
        "frames_started": 0,
        "laser_tags_seen": 0,
        "detector_tags_seen": 0,
        "pixel_tags_seen": 0,
        "detector_tags_assigned": 0,
        "detector_tags_without_laser": 0,
        "overflow_begin": 0,
        "overflow_end": 0,
        "missed_events": 0,
        "error_events": 0,
        "buffer_chunks_full": 0,
    }

    armed = False
    pixel_lists = None
    frame_start_ps = None
    frame_stop_ps = None
    last_laser_time_ps = None
    finalize_deadline = None

    try:
        tagger.setTriggerLevel(laser_ch, laser_trigger_v)
        tagger.setTriggerLevel(detector_ch, detector_trigger_v)
        tagger.setTriggerLevel(pixel_ch, pixel_trigger_v)

        adjusted_laser_delay_ps = int(laser_input_delay_ps)
        if use_conditional_filter:
            # This may distort the timing relationship between laser and detector
            # tags in the raw stream. Disable if the hardware stream looks wrong.
            tagger.setConditionalFilter(trigger=[detector_ch], filtered=[laser_ch])
        if int(laser_event_divider) > 1:
            tagger.setEventDivider(laser_ch, int(laser_event_divider))

        if adjusted_laser_delay_ps:
            tagger.setInputDelay(laser_ch, adjusted_laser_delay_ps)
        if detector_input_delay_ps:
            tagger.setInputDelay(detector_ch, int(detector_input_delay_ps))
        if pixel_input_delay_ps:
            tagger.setInputDelay(pixel_ch, int(pixel_input_delay_ps))

        stream = TimeTagger.TimeTagStream(
            tagger=tagger,
            n_max_events=stream_buffer_size,
            channels=[laser_ch, detector_ch, pixel_ch],
        )
        stream.start()

        while len(frames) < n_frames:
            data = stream.getData()

            if data.size == stream_buffer_size:
                stats["buffer_chunks_full"] += 1

            if data.size > 0:
                event_types = data.getEventTypes()
                non_time_mask = event_types != 0
                if np.any(non_time_mask):
                    stats["overflow_begin"] += int(np.count_nonzero(event_types == 2))
                    stats["overflow_end"] += int(np.count_nonzero(event_types == 3))
                    stats["error_events"] += int(np.count_nonzero(event_types == 1))
                    missed_events = data.getMissedEvents()
                    stats["missed_events"] += int(np.sum(missed_events[event_types == 4]))

                valid_mask = event_types == 0
                if np.any(valid_mask):
                    timestamps = data.getTimestamps()[valid_mask].astype(np.int64, copy=False)
                    channels = data.getChannels()[valid_mask].astype(np.int64, copy=False)

                    if not armed:
                        pixel_positions = np.flatnonzero(channels == pixel_ch)
                        if pixel_positions.size:
                            arm_position = int(pixel_positions[0])
                            pre_arm_lasers = timestamps[:arm_position][channels[:arm_position] == laser_ch]
                            if pre_arm_lasers.size:
                                last_laser_time_ps = int(pre_arm_lasers[-1])

                            timestamps = timestamps[arm_position:]
                            channels = channels[arm_position:]
                            armed = True
                            pixel_lists = new_pixel_lists()
                            frame_start_ps = int(timestamps[0])
                            frame_stop_ps = frame_start_ps + frame_duration_ps
                            finalize_deadline = time.monotonic() + (
                                frame_duration_ps / 1e12
                            ) + final_pixel_margin_s
                            stats["frames_started"] += 1

                    if armed and timestamps.size:
                        laser_ts = timestamps[channels == laser_ch]
                        detector_ts = timestamps[channels == detector_ch]
                        pixel_ts_all = timestamps[channels == pixel_ch]

                        stats["laser_tags_seen"] += int(laser_ts.size)
                        stats["detector_tags_seen"] += int(detector_ts.size)
                        stats["pixel_tags_seen"] += int(pixel_ts_all.size)

                        if frame_stop_ps is not None:
                            laser_ts = laser_ts[laser_ts < frame_stop_ps]
                            detector_ts = detector_ts[detector_ts < frame_stop_ps]

                        if detector_ts.size:
                            if laser_ts.size:
                                if last_laser_time_ps is None:
                                    laser_reference = laser_ts
                                else:
                                    laser_reference = np.empty(laser_ts.size + 1, dtype=np.int64)
                                    laser_reference[0] = last_laser_time_ps
                                    laser_reference[1:] = laser_ts

                                laser_indices = np.searchsorted(
                                    laser_reference, detector_ts, side="right"
                                ) - 1
                                valid_detector_mask = laser_indices >= 0
                                valid_detector_ts = detector_ts[valid_detector_mask]
                                detector_delays_ps = (
                                    valid_detector_ts - laser_reference[laser_indices[valid_detector_mask]]
                                )
                            elif last_laser_time_ps is not None:
                                valid_detector_ts = detector_ts
                                detector_delays_ps = detector_ts - last_laser_time_ps
                                valid_detector_mask = np.ones(detector_ts.shape, dtype=bool)
                            else:
                                valid_detector_ts = detector_ts[:0]
                                detector_delays_ps = detector_ts[:0]
                                valid_detector_mask = np.zeros(detector_ts.shape, dtype=bool)

                            stats["detector_tags_without_laser"] += int(
                                detector_ts.size - np.count_nonzero(valid_detector_mask)
                            )

                            if valid_detector_ts.size:
                                detector_pixel_indices = (
                                    (valid_detector_ts - frame_start_ps) // int(pixel_dwell_ps)
                                ).astype(np.int64, copy=False)
                                in_frame_mask = (
                                    (detector_pixel_indices >= 0)
                                    & (detector_pixel_indices < total_pixels)
                                )

                                if np.any(in_frame_mask):
                                    assigned_pixels = detector_pixel_indices[in_frame_mask]
                                    assigned_delays = detector_delays_ps[in_frame_mask]
                                    order = np.argsort(assigned_pixels, kind="stable")
                                    assigned_pixels = assigned_pixels[order]
                                    assigned_delays = assigned_delays[order]
                                    split_points = np.flatnonzero(np.diff(assigned_pixels)) + 1

                                    for pixel_index_group, delay_group in zip(
                                        np.split(assigned_pixels, split_points),
                                        np.split(assigned_delays, split_points),
                                    ):
                                        pixel_lists[int(pixel_index_group[0])].extend(
                                            delay_group.tolist()
                                        )

                                    stats["detector_tags_assigned"] += int(assigned_delays.size)

                        if laser_ts.size:
                            last_laser_time_ps = int(laser_ts[-1])

                        crossed_frame_end = bool(np.any(timestamps >= frame_stop_ps))
                        if crossed_frame_end or time.monotonic() >= finalize_deadline:
                            frames.append(finalize_frame(pixel_lists))
                            armed = False
                            pixel_lists = None
                            frame_start_ps = None
                            frame_stop_ps = None
                            last_laser_time_ps = None
                            finalize_deadline = None

                            if len(frames) >= n_frames:
                                break

            if armed and finalize_deadline is not None and time.monotonic() >= finalize_deadline:
                frames.append(finalize_frame(pixel_lists))
                armed = False
                pixel_lists = None
                frame_start_ps = None
                frame_stop_ps = None
                last_laser_time_ps = None
                finalize_deadline = None

                if len(frames) >= n_frames:
                    break

            if data.size == 0:
                time.sleep(poll_sleep_s)

        return frames, stats

    finally:
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
        TimeTagger.freeTimeTagger(tagger)


def save_raw_flim_data(frames, output_dir, run_name, acquisition_parameters):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = output_dir / f"{timestamp}_{run_name}"
    frame_stack = np.asarray(frames, dtype=object)
    laser_period_ps = int(round(1e12 / acquisition_parameters["laser_frequency_hz"]))

    intensity = np.zeros((len(frame_stack), frame_stack.shape[1], frame_stack.shape[2]), dtype=np.uint32)
    mean_delay_ps = np.full((len(frame_stack), frame_stack.shape[1], frame_stack.shape[2]), np.nan, dtype=np.float32)

    for frame_index, frame in enumerate(frame_stack):
        for y_index in range(frame.shape[0]):
            for x_index in range(frame.shape[1]):
                delays = frame[y_index, x_index]
                intensity[frame_index, y_index, x_index] = len(delays)
                if len(delays):
                    mean_delay_ps[frame_index, y_index, x_index] = float(
                        np.mean(np.mod(delays, laser_period_ps))
                    )

    np.savez_compressed(
        stem.with_name(f"{stem.name}_raw.npz"),
        frames=frame_stack,
        acquisition_parameters=np.asarray(acquisition_parameters, dtype=object),
    )
    tifffile.imwrite(stem.with_name(f"{stem.name}_intensity.tiff"), intensity)
    tifffile.imwrite(stem.with_name(f"{stem.name}_mean_delay_ps.tiff"), mean_delay_ps)
    stem.with_name(f"{stem.name}_metadata.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "run_name": run_name,
                "acquisition_parameters": acquisition_parameters,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return stem


if __name__ == "__main__":
    run_name = "flim_test_001"
    output_dir = Path("data") / "swabian_raw_flim"

    x_pixels = 512
    y_pixels = 512
    extra_left = 30
    extra_right = 10
    n_frames = 1
    pixel_dwell_ps = int(10e6)
    laser_frequency_hz = 80e6

    laser_ch = 1
    detector_ch = 2
    pixel_ch = 3

    laser_trigger_v = 0.05
    detector_trigger_v = 0.2
    pixel_trigger_v = 0.2

    laser_input_delay_ps = 0
    detector_input_delay_ps = 0
    pixel_input_delay_ps = 0
    laser_event_divider = 100
    use_conditional_filter = False

    acquisition_parameters = {
        "x_pixels": x_pixels,
        "y_pixels": y_pixels,
        "extra_left": extra_left,
        "extra_right": extra_right,
        "n_frames": n_frames,
        "pixel_dwell_ps": pixel_dwell_ps,
        "laser_frequency_hz": laser_frequency_hz,
        "laser_ch": laser_ch,
        "detector_ch": detector_ch,
        "pixel_ch": pixel_ch,
        "laser_trigger_v": laser_trigger_v,
        "detector_trigger_v": detector_trigger_v,
        "pixel_trigger_v": pixel_trigger_v,
        "laser_input_delay_ps": laser_input_delay_ps,
        "detector_input_delay_ps": detector_input_delay_ps,
        "pixel_input_delay_ps": pixel_input_delay_ps,
        "laser_event_divider": laser_event_divider,
        "use_conditional_filter": use_conditional_filter,
    }

    print("Starting raw FLIM acquisition.")
    print("Run this script first, then start the DAQ-side scan.")

    start_time = time.perf_counter()
    frames, stats = acquire_raw_flim(
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        n_frames=n_frames,
        pixel_dwell_ps=pixel_dwell_ps,
        extra_left=extra_left,
        extra_right=extra_right,
        laser_frequency_hz=laser_frequency_hz,
        laser_ch=laser_ch,
        detector_ch=detector_ch,
        pixel_ch=pixel_ch,
        laser_trigger_v=laser_trigger_v,
        detector_trigger_v=detector_trigger_v,
        pixel_trigger_v=pixel_trigger_v,
        laser_input_delay_ps=laser_input_delay_ps,
        detector_input_delay_ps=detector_input_delay_ps,
        pixel_input_delay_ps=pixel_input_delay_ps,
        laser_event_divider=laser_event_divider,
        use_conditional_filter=use_conditional_filter,
    )
    elapsed_s = time.perf_counter() - start_time

    stem = save_raw_flim_data(
        frames=frames,
        output_dir=output_dir,
        run_name=run_name,
        acquisition_parameters=acquisition_parameters,
    )

    print(f"Finished in {elapsed_s:.3f} s")
    print(json.dumps(stats, indent=2))
    print(f"Saved raw data to {stem.with_name(f'{stem.name}_raw.npz')}")
