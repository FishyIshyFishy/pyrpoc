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
    use_conditional_filter=True,
    stream_buffer_size=1_000_000,
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

    total_x_pixels = int(x_pixels) + int(extra_left) + int(extra_right)
    total_pixels = total_x_pixels * int(y_pixels)
    laser_period_ps = int(round(1e12 / float(laser_frequency_hz)))

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

    pixel_lists = new_pixel_lists()
    current_pixel_index = -1
    pixel_count_in_frame = 0
    current_pixel_start_ps = None
    current_pixel_end_ps = None
    last_laser_time_ps = None
    finalize_deadline = None

    try:
        tagger.setTriggerLevel(laser_ch, laser_trigger_v)
        tagger.setTriggerLevel(detector_ch, detector_trigger_v)
        tagger.setTriggerLevel(pixel_ch, pixel_trigger_v)

        adjusted_laser_delay_ps = int(laser_input_delay_ps)
        if use_conditional_filter:
            adjusted_laser_delay_ps -= laser_period_ps
            tagger.setConditionalFilter(trigger=[detector_ch], filtered=[laser_ch])

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
                raise RuntimeError(
                    "TimeTagStream buffer filled completely. Increase stream_buffer_size."
                )

            if data.size > 0:
                event_types = data.getEventTypes()
                if np.any(event_types != 0):
                    raise RuntimeError(
                        f"Received non-TimeTag events: {np.unique(event_types[event_types != 0]).tolist()}"
                    )

                for timestamp_ps, channel in zip(data.getTimestamps(), data.getChannels()):
                    timestamp_ps = int(timestamp_ps)
                    channel = int(channel)

                    if channel == pixel_ch:
                        if pixel_count_in_frame == total_pixels:
                            frames.append(finalize_frame(pixel_lists))
                            if len(frames) >= n_frames:
                                break

                            pixel_lists = new_pixel_lists()
                            current_pixel_index = -1
                            pixel_count_in_frame = 0
                            current_pixel_start_ps = None
                            current_pixel_end_ps = None
                            last_laser_time_ps = None
                            finalize_deadline = None

                        current_pixel_index += 1
                        pixel_count_in_frame += 1
                        current_pixel_start_ps = timestamp_ps
                        current_pixel_end_ps = timestamp_ps + int(pixel_dwell_ps)

                        if pixel_count_in_frame == total_pixels:
                            finalize_deadline = time.monotonic() + (
                                pixel_dwell_ps / 1e12
                            ) + final_pixel_margin_s

                    elif channel == laser_ch:
                        last_laser_time_ps = timestamp_ps

                    elif channel == detector_ch:
                        if current_pixel_index < 0 or current_pixel_index >= total_pixels:
                            continue
                        if last_laser_time_ps is None:
                            continue
                        if current_pixel_start_ps is None or current_pixel_end_ps is None:
                            continue
                        if timestamp_ps < current_pixel_start_ps:
                            continue
                        if timestamp_ps >= current_pixel_end_ps:
                            continue
                        if timestamp_ps < last_laser_time_ps:
                            continue

                        pixel_lists[current_pixel_index].append(timestamp_ps - last_laser_time_ps)

                if len(frames) >= n_frames:
                    break

            if pixel_count_in_frame == total_pixels and finalize_deadline is not None:
                if time.monotonic() >= finalize_deadline:
                    frames.append(finalize_frame(pixel_lists))
                    if len(frames) >= n_frames:
                        break

                    pixel_lists = new_pixel_lists()
                    current_pixel_index = -1
                    pixel_count_in_frame = 0
                    current_pixel_start_ps = None
                    current_pixel_end_ps = None
                    last_laser_time_ps = None
                    finalize_deadline = None

            if data.size == 0:
                time.sleep(poll_sleep_s)

        return frames

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

    intensity = np.zeros((len(frame_stack), frame_stack.shape[1], frame_stack.shape[2]), dtype=np.uint32)
    mean_delay_ps = np.full((len(frame_stack), frame_stack.shape[1], frame_stack.shape[2]), np.nan, dtype=np.float32)

    for frame_index, frame in enumerate(frame_stack):
        for y_index in range(frame.shape[0]):
            for x_index in range(frame.shape[1]):
                delays = frame[y_index, x_index]
                intensity[frame_index, y_index, x_index] = len(delays)
                if len(delays):
                    mean_delay_ps[frame_index, y_index, x_index] = float(np.mean(delays))

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

    x_pixels = 542
    y_pixels = 512
    extra_left = 50
    extra_right = 50
    n_frames = 1
    pixel_dwell_ps = int(10e6)
    laser_frequency_hz = 80e6

    laser_ch = 1
    detector_ch = 2
    pixel_ch = 3

    laser_trigger_v = 0.2
    detector_trigger_v = 0.2
    pixel_trigger_v = 0.2

    laser_input_delay_ps = 0
    detector_input_delay_ps = 0
    pixel_input_delay_ps = 0
    use_conditional_filter = True

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
        "use_conditional_filter": use_conditional_filter,
    }

    print("Starting raw FLIM acquisition.")
    print("Run this script first, then start the DAQ-side scan.")

    start_time = time.perf_counter()
    frames = acquire_raw_flim(
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
    print(f"Saved raw data to {stem.with_name(f'{stem.name}_raw.npz')}")
