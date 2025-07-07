import numpy as np
import nidaqmx
import abc
from pyrpoc.instruments.instrument_manager import *
import time
import nidaqmx
from nidaqmx.constants import AcquisitionType
import tifffile
from pathlib import Path
from .base_acquisition import Acquisition


class SplitDataStream(Acquisition):
    def __init__(self, galvo=None, data_inputs=None, prior_stage=None, num_frames=1, split_percentage=50, signal_bus=None, acquisition_parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.galvo = galvo
        self.data_inputs = data_inputs or []
        self.prior_stage = prior_stage
        self.num_frames = num_frames
        self.split_percentage = split_percentage
        self.signal_bus = signal_bus
        self.verified = False
        self.acquisition_parameters = acquisition_parameters or {}
        
        if self.galvo is None:
            raise ValueError("Galvo is required for SplitDataStream acquisition")
        if not self.data_inputs:
            raise ValueError("At least one DataInput is required for SplitDataStream acquisition")
        if self.prior_stage is None:
            raise ValueError("Prior stage is required for SplitDataStream acquisition")
        
        self.rpoc_enabled = False
        self.rpoc_masks = {}
        self.rpoc_channels = {}

    def configure_rpoc(self, rpoc_enabled, rpoc_masks=None, rpoc_channels=None, **kwargs):
        self.rpoc_enabled = rpoc_enabled
        if rpoc_masks:
            self.rpoc_masks = rpoc_masks
        if rpoc_channels:
            self.rpoc_channels = rpoc_channels
        
        if self.signal_bus:
            self.signal_bus.console_message.emit(f"RPOC Configured - enabled: {rpoc_enabled}, masks: {len(rpoc_masks) if rpoc_masks else 0}, channels: {len(rpoc_channels) if rpoc_channels else 0}")

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

        numtiles_x = self.acquisition_parameters.get('numtiles_x', 1)
        numtiles_y = self.acquisition_parameters.get('numtiles_y', 1)
        numtiles_z = self.acquisition_parameters.get('numtiles_z', 1)
        tile_size_x = self.acquisition_parameters.get('tile_size_x', 100)
        tile_size_y = self.acquisition_parameters.get('tile_size_y', 100)
        tile_size_z = self.acquisition_parameters.get('tile_size_z', 50)
        
        try:
            start_x, start_y = self.prior_stage.get_xy()
            start_z = self.prior_stage.get_z()
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error getting current stage position: {e}")
            raise RuntimeError(f"Failed to get current stage position: {e}")
        
        stage_positions = []
        tile_indices = []
        for z_idx in range(numtiles_z):
            for y_idx in range(numtiles_y):
                for x_idx in range(numtiles_x):
                    x_pos = start_x + x_idx * tile_size_x
                    y_pos = start_y + y_idx * tile_size_y
                    z_pos = start_z + z_idx * tile_size_z
                    stage_positions.append((x_pos, y_pos, z_pos))
                    tile_indices.append((x_idx, y_idx, z_idx))

        all_frames = []
        total_positions = len(stage_positions) * self.num_frames
        current_position = 0
        
        for frame_idx in range(self.num_frames):
            if self._stop_flag and self._stop_flag():
                break
            
            for pos_idx, (x_pos, y_pos, z_pos) in enumerate(stage_positions):
                if self._stop_flag and self._stop_flag():
                    break
                
                try:
                    if self.signal_bus:
                        self.signal_bus.console_message.emit(f"Moving to position {pos_idx + 1}/{len(stage_positions)}: X={x_pos}, Y={y_pos}, Z={z_pos}")
                    
                    self.prior_stage.move_xy(x_pos, y_pos)
                    self.prior_stage.move_z(z_pos)
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    if self.signal_bus:
                        self.signal_bus.console_message.emit(f"Error moving stage: {e}")
                    raise RuntimeError(f"Failed to move stage to position {pos_idx + 1}: {e}")

                frame_data = self.collect_split_data(self.galvo, ai_channels)
                
                if self.signal_bus:
                    self.signal_bus.data_signal.emit(frame_data, current_position, total_positions, False)
                all_frames.append(frame_data)
                current_position += 1
        
        if all_frames:
            final_data = np.stack(all_frames)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, len(all_frames)-1, total_positions, True)
            
            self.metadata['tile_order'] = tile_indices            
            self.save_data(final_data)
            return final_data
        else:
            return None
    
    def collect_split_data(self, galvo, ai_channels):
        try:
            rate = galvo.parameters.get('sample_rate', 1000000)
            dwell_time = self.acquisition_parameters.get('dwell_time', 10e-6)
            extra_left = self.acquisition_parameters.get('extrasteps_left', 50)
            extra_right = self.acquisition_parameters.get('extrasteps_right', 50)
            numsteps_x = self.acquisition_parameters.get('x_pixels', 512)
            numsteps_y = self.acquisition_parameters.get('y_pixels', 512)
            slow_channel = galvo.parameters.get('slow_axis_channel', 0)
            fast_channel = galvo.parameters.get('fast_axis_channel', 1)
            device_name = galvo.parameters.get('device_name', 'Dev1')

            pixel_samples = max(1, int(dwell_time * rate))
            total_x = numsteps_x + extra_left + extra_right
            total_y = numsteps_y
            total_samples = total_x * total_y * pixel_samples

            waveform = galvo.generate_raster_waveform(self.acquisition_parameters)
            
            rpoc_do_channels = []
            rpoc_ttl_signals = []
            
            if self.rpoc_enabled and self.rpoc_masks and self.rpoc_channels:
                for channel_id, mask in self.rpoc_masks.items():
                    if channel_id in self.rpoc_channels:
                        channel_info = self.rpoc_channels[channel_id]
                        device = channel_info.get('device', device_name)
                        port_line = channel_info.get('port_line', f'port0/line{4+channel_id-1}')
                        
                        if isinstance(mask, np.ndarray):
                            mask_array = mask
                        else:
                            try:
                                from PIL import Image
                                if isinstance(mask, Image.Image):
                                    mask_array = np.array(mask)
                                else:
                                    mask_array = mask
                            except:
                                mask_array = mask
                        
                        mask_array = mask_array > 0
                        
                        padded_mask = []
                        for row in range(numsteps_y):
                            padded_row = np.concatenate((
                                np.zeros(extra_left, dtype=bool),
                                mask_array[row, :] if row < mask_array.shape[0] else np.zeros(numsteps_x, dtype=bool),
                                np.zeros(extra_right, dtype=bool)
                            ))
                            padded_mask.append(padded_row)
                        
                        flat_mask = np.repeat(np.array(padded_mask).ravel(), pixel_samples).astype(bool)
                        
                        rpoc_do_channels.append(f"{device}/{port_line}")
                        rpoc_ttl_signals.append(flat_mask)
            
            with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
                ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_channel}")
                ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_channel}")
                
                for ch in ai_channels:
                    ai_task.ai_channels.add_ai_voltage_chan(ch)

                ao_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=total_samples
                )

                ai_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    source=f"/{device_name}/ao/SampleClock",
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=total_samples
                )

                do_task = None
                if rpoc_do_channels and rpoc_ttl_signals:
                    do_task = nidaqmx.Task()
                    
                    if len(rpoc_do_channels) == 1:
                        do_task.do_channels.add_do_chan(rpoc_do_channels[0])
                        do_task.timing.cfg_samp_clk_timing(
                            rate=rate,
                            source=f"/{device_name}/ao/SampleClock",
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=total_samples
                        )
                        do_task.write(rpoc_ttl_signals[0].tolist(), auto_start=False)
                    else:
                        for chan in rpoc_do_channels:
                            do_task.do_channels.add_do_chan(chan)
                        do_task.timing.cfg_samp_clk_timing(
                            rate=rate,
                            source=f"/{device_name}/ao/SampleClock",
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=total_samples
                        )
                        data_to_write = [sig.tolist() for sig in rpoc_ttl_signals]
                        do_task.write(data_to_write, auto_start=False)

                ao_task.write(waveform.T, auto_start=False)
                
                ai_task.start()
                if do_task:
                    do_task.start()
                ao_task.start()

                timeout = total_samples / rate + 5
                ao_task.wait_until_done(timeout=timeout)
                ai_task.wait_until_done(timeout=timeout)
                if do_task:
                    do_task.wait_until_done(timeout=timeout)
                
                acq_data = np.array(ai_task.read(number_of_samples_per_channel=total_samples))
                
                input_results = []
                for i in range(len(ai_channels)):
                    channel_data = acq_data if len(ai_channels) == 1 else acq_data[i]
                    reshaped = channel_data.reshape(total_y, total_x, pixel_samples)
                    input_results.append(reshaped)
                
                split_point = int(self.split_percentage / 100.0 * pixel_samples)
                
                if self.signal_bus:
                    self.signal_bus.console_message.emit(f"Split Data Stream: {pixel_samples} samples per pixel, split at {split_point} samples ({self.split_percentage}%)")
                    self.signal_bus.console_message.emit(f"Channel 1: samples 0-{split_point-1}, Channel 2: samples {split_point}-{pixel_samples-1}, Channel 3: all {pixel_samples} samples")
                
                output_channels = []
                
                for input_channel in input_results:
                    height, width, samples_per_pixel = input_channel.shape
                    
                    first_portion = np.zeros((height, width))
                    for y in range(height):
                        for x in range(width):
                            pixel_samples_data = input_channel[y, x, :]
                            first_portion[y, x] = np.mean(pixel_samples_data[:split_point])
                    
                    second_portion = np.zeros((height, width))
                    for y in range(height):
                        for x in range(width):
                            pixel_samples_data = input_channel[y, x, :]
                            second_portion[y, x] = np.mean(pixel_samples_data[split_point:])
                    
                    full_data = np.mean(input_channel, axis=2)
                    
                    output_channels.extend([first_portion, second_portion, full_data])
                
                if len(output_channels) == 1:
                    return output_channels[0]
                else:
                    return np.stack(output_channels)
                    
        except Exception as e:
            self.signal_bus.console_message.emit(f"Error in DAQ acquisition: {e}")
    
    def save_data(self, data):
        if not self.save_enabled or not self.save_path:
            return
        
        try:
            save_dir = Path(self.save_path).parent
            if not save_dir.is_dir():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            if len(data.shape) == 4:
                num_total_images, num_channels, height, width = data.shape
            elif len(data.shape) == 3:
                num_total_images, height, width = data.shape
                num_channels = 1
                data = data.reshape(num_total_images, 1, height, width)
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
            
            num_input_channels = num_channels // 3
            if num_channels % 3 != 0:
                raise ValueError(f"Split stream data should have 3 channels per input, got {num_channels} total channels")
            
            input_channel_names = self._get_input_channel_names()
            
            for input_ch_idx in range(num_input_channels):
                input_channel_name = input_channel_names.get(input_ch_idx, f"input{input_ch_idx}")
                
                first_ch_idx = input_ch_idx * 3
                second_ch_idx = input_ch_idx * 3 + 1
                full_ch_idx = input_ch_idx * 3 + 2
  
                first_data = data[:, first_ch_idx, :, :]
                first_filename = f"{Path(self.save_path).stem}_{input_channel_name}_first.tiff"
                first_filepath = save_dir / first_filename
                tifffile.imwrite(first_filepath, first_data)
                
                second_data = data[:, second_ch_idx, :, :]
                second_filename = f"{Path(self.save_path).stem}_{input_channel_name}_second.tiff"
                second_filepath = save_dir / second_filename
                tifffile.imwrite(second_filepath, second_data)
                
                full_data = data[:, full_ch_idx, :, :]
                full_filename = f"{Path(self.save_path).stem}_{input_channel_name}_full.tiff"
                full_filepath = save_dir / full_filename
                tifffile.imwrite(full_filepath, full_data)
                
                if self.signal_bus:
                    self.signal_bus.console_message.emit(f"Saved split stream data for {input_channel_name}: first, second, and full channels")
            
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving split stream data: {e}")
            else:
                print(f"Error saving split stream data: {e}")
    
    def _get_input_channel_names(self):

        input_channel_names = {}
        input_ch_idx = 0
        
        if hasattr(self, 'data_inputs') and self.data_inputs:
            for data_input in self.data_inputs:
                if hasattr(data_input, 'parameters'):
                    input_channels = data_input.parameters.get('input_channels', [])
                    channel_names_param = data_input.parameters.get('channel_names', {})
                    
                    for ch in input_channels:
                        channel_name = channel_names_param.get(str(ch), f"ch{ch}")
                        input_channel_names[input_ch_idx] = channel_name
                        input_ch_idx += 1
        
        if not input_channel_names:
            for ch in range(input_ch_idx):
                input_channel_names[ch] = f"input{ch}"
        
        return input_channel_names