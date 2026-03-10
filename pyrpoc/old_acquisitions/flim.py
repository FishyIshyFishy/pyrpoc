import time
from pathlib import Path

import nidaqmx
import numpy as np
import tifffile
from nidaqmx.constants import AcquisitionType, LineGrouping

from .base_acquisition import Acquisition


class FLIM(Acquisition):
    def __init__(self, galvo=None, data_inputs=None, num_frames=1, signal_bus=None, acquisition_parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.galvo = galvo
        self.data_inputs = data_inputs or []
        self.num_frames = num_frames
        self.signal_bus = signal_bus
        self.verified = False
        self.acquisition_parameters = acquisition_parameters or {}

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        # FLIM no longer carries RPOC-specific acquisition behavior.
        return None

    def perform_acquisition(self):
        self.save_metadata()

        ai_channels = []
        for data_input in self.data_inputs:
            channels = data_input.parameters.get('input_channels', [])
            device = data_input.parameters.get('device_name', 'Dev1')
            for ch in channels:
                ai_channels.append(f"{device}/ai{ch}")

        if not ai_channels:
            ai_channels = ["Dev1/ai0"]

        all_frames = []
        for frame_idx in range(self.num_frames):
            if self._stop_flag and self._stop_flag():
                break

            frame_data = self.collect_data(self.galvo, ai_channels)

            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame_idx, self.num_frames, False)
            all_frames.append(frame_data)

        if all_frames:
            final_data = np.stack(all_frames)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, len(all_frames) - 1, self.num_frames, True)
            self.save_data(final_data)
            return final_data
        return None

    def generate_simulated_confocal(self):
        x_pixels = self.acquisition_parameters.get('x_pixels', 512)
        y_pixels = self.acquisition_parameters.get('y_pixels', 512)

        num_channels = 1
        if self.data_inputs:
            total_channels = 0
            for data_input in self.data_inputs:
                channels = data_input.parameters.get('input_channels', [])
                total_channels += len(channels)
            num_channels = max(1, total_channels)

        frame = np.zeros((num_channels, y_pixels, x_pixels))

        for ch in range(num_channels):
            num_circles = np.random.randint(2, 6)
            for _ in range(num_circles):
                center_x = np.random.randint(50, x_pixels - 50)
                center_y = np.random.randint(50, y_pixels - 50)
                radius = np.random.randint(10, 30)
                intensity = np.random.uniform(0.3, 1.0)

                y, x = np.ogrid[:y_pixels, :x_pixels]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
                frame[ch, mask] += intensity

            frame[ch] += np.random.normal(0, 0.1, frame[ch].shape)
            frame[ch] = np.clip(frame[ch], 0, 1)

        time.sleep(1)
        return frame

    def _build_pixel_start_signal(self, total_y, total_x, pixel_samples, extra_left, numsteps_x):
        if not self.acquisition_parameters.get('pixel_start_enabled', False):
            return None

        pulse_samples = int(self.acquisition_parameters.get('pixel_start_pulse_samples', 1))
        pulse_samples = max(1, min(pixel_samples, pulse_samples))

        ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)
        active_start = extra_left
        active_stop = extra_left + numsteps_x
        ttl[:, active_start:active_stop, :pulse_samples] = True
        return ttl.ravel()

    def _get_pixel_start_line(self):
        if not self.acquisition_parameters.get('pixel_start_enabled', False):
            return None

        device_name = self.acquisition_parameters.get(
            'pixel_start_device_name',
            self.galvo.parameters.get('device_name', 'Dev1'),
        )
        port = self.acquisition_parameters.get('pixel_start_port', 'port0')
        line = self.acquisition_parameters.get('pixel_start_line', 0)
        return f"{device_name}/{port}/line{line}"

    def collect_data(self, galvo, ai_channels):
        try:
            rate = galvo.parameters.get('sample_rate', 1000000)
            dwell_time = self.acquisition_parameters.get('dwell_time', 10)
            dwell_time_sec = dwell_time / 1e6
            extra_left = self.acquisition_parameters.get('extrasteps_left', 50)
            extra_right = self.acquisition_parameters.get('extrasteps_right', 50)
            numsteps_x = self.acquisition_parameters.get('x_pixels', 512)
            numsteps_y = self.acquisition_parameters.get('y_pixels', 512)
            slow_channel = galvo.parameters.get('slow_axis_channel', 0)
            fast_channel = galvo.parameters.get('fast_axis_channel', 1)
            device_name = galvo.parameters.get('device_name', 'Dev1')

            pixel_samples = max(1, int(dwell_time_sec * rate))
            total_x = numsteps_x + extra_left + extra_right
            total_y = numsteps_y
            total_samples = total_x * total_y * pixel_samples

            waveform = galvo.generate_raster_waveform(self.acquisition_parameters)
            pixel_start_signal = self._build_pixel_start_signal(
                total_y=total_y,
                total_x=total_x,
                pixel_samples=pixel_samples,
                extra_left=extra_left,
                numsteps_x=numsteps_x,
            )
            pixel_start_line = self._get_pixel_start_line()

            with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
                ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_channel}")
                ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_channel}")
                for ch in ai_channels:
                    ai_task.ai_channels.add_ai_voltage_chan(ch)

                ao_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=total_samples,
                )
                ai_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    source=f"/{device_name}/ao/SampleClock",
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=total_samples,
                )

                pixel_start_task = None
                if pixel_start_signal is not None and pixel_start_line is not None:
                    pixel_start_task = nidaqmx.Task()
                    pixel_start_task.do_channels.add_do_chan(
                        pixel_start_line,
                        line_grouping=LineGrouping.CHAN_PER_LINE,
                    )
                    pixel_start_task.timing.cfg_samp_clk_timing(
                        rate=rate,
                        source=f"/{device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=total_samples,
                    )
                    pixel_start_task.write(pixel_start_signal.tolist(), auto_start=False)

                ao_task.write(waveform, auto_start=False)
                ai_task.start()
                if pixel_start_task:
                    pixel_start_task.start()
                ao_task.start()

                timeout = total_samples / rate + 5
                ao_task.wait_until_done(timeout=timeout)
                ai_task.wait_until_done(timeout=timeout)
                if pixel_start_task:
                    pixel_start_task.wait_until_done(timeout=timeout)
                    pixel_start_task.close()

                acq_data = np.array(ai_task.read(number_of_samples_per_channel=total_samples))

                results = []
                for i in range(len(ai_channels)):
                    channel_data = acq_data if len(ai_channels) == 1 else acq_data[i]
                    reshaped = channel_data.reshape(total_y, total_x, pixel_samples)
                    pixel_values = np.mean(reshaped, axis=2)
                    cropped = pixel_values[:, extra_left:extra_left + numsteps_x]
                    results.append(cropped)

                if len(results) == 1:
                    return results[0]
                return np.stack(results)

        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f'Error in DAQ acquisition: {e}')
            return self.generate_simulated_confocal()

    def save_data(self, data):
        if not self.save_enabled or not self.save_path:
            return

        try:
            save_dir = Path(self.save_path).parent
            if not save_dir.is_dir():
                save_dir.mkdir(parents=True, exist_ok=True)

            if len(data.shape) == 4:
                num_frames, num_channels, height, width = data.shape
            elif len(data.shape) == 3:
                num_frames, height, width = data.shape
                num_channels = 1
                data = data.reshape(num_frames, 1, height, width)
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

            channel_names = self._get_channel_names()

            for ch in range(num_channels):
                channel_data = data[:, ch, :, :]
                channel_name = channel_names.get(ch, f"ch{ch:02d}")
                filename = f"{Path(self.save_path).stem}_{channel_name}.tiff"
                filepath = save_dir / filename
                tifffile.imwrite(filepath, channel_data)

        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving FLIM data: {e}")

    def _get_channel_names(self):
        channel_names = {}
        channel_idx = 0

        if hasattr(self, 'data_inputs') and self.data_inputs:
            for data_input in self.data_inputs:
                if hasattr(data_input, 'parameters'):
                    input_channels = data_input.parameters.get('input_channels', [])
                    channel_names_param = data_input.parameters.get('channel_names', {})

                    for ch in input_channels:
                        channel_name = channel_names_param.get(str(ch), f"ch{ch}")
                        channel_names[channel_idx] = channel_name
                        channel_idx += 1

        if not channel_names:
            for ch in range(channel_idx):
                channel_names[ch] = f"ch{ch:02d}"

        return channel_names
