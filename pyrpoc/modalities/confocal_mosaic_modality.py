from .base import BaseModality, Acquisition
from pyrpoc.displays.multichan_tiled import MultichannelImageDisplayWidget
from typing import List, Type, Dict, Any
import numpy as np
import time
from pathlib import Path
import tifffile
import nidaqmx
from nidaqmx.constants import AcquisitionType
from pyrpoc.instruments.instrument_manager import *

class ConfocalMosaicModality(BaseModality):
    @property
    def name(self) -> str:
        return "Confocal mosaic"
    
    @property
    def key(self) -> str:
        return "confocal mosaic"
    
    @property
    def required_instruments(self) -> List[str]:
        return ["galvo", "data input", "prior stage"]
    
    @property
    def compatible_displays(self):
        return [MultichannelImageDisplayWidget]
    
    @property
    def parameter_groups(self) -> Dict[str, List[str]]:
        return {
            'Image Dimensions': ['x_pixels', 'y_pixels'],
            'Scanning': ['dwell_time', 'extrasteps_left', 'extrasteps_right'],
            'Galvo Control': ['amplitude_x', 'amplitude_y', 'offset_x', 'offset_y'],
            'Tiling': ['numtiles_x', 'numtiles_y', 'numtiles_z'],
            'Tile Sizes': ['tile_size_x', 'tile_size_y', 'tile_size_z']
        }
    
    @property
    def required_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'dwell_time': {
                'type': 'float',
                'default': 10.0,
                'range': (1.0, 1000.0),
                'unit': 'μs',
                'description': 'Per pixel dwell time'
            },
            'extrasteps_left': {
                'type': 'int',  
                'default': 50,
                'range': (0, 10000),
                'description': 'Extra steps left in fast direction'
            },
            'extrasteps_right': {
                'type': 'int',
                'default': 50,
                'range': (0, 10000),
                'description': 'Extra steps right in fast direction'
            },
            'amplitude_x': {
                'type': 'float',
                'default': 0.5,
                'range': (0.0, 10.0),
                'unit': 'V',
                'description': 'Amplitude for X axis'
            },
            'amplitude_y': {
                'type': 'float',
                'default': 0.5,
                'range': (0.0, 10.0),
                'unit': 'V',
                'description': 'Amplitude for Y axis'
            },
            'offset_x': {
                'type': 'float',
                'default': 0.0,
                'range': (-10.0, 10.0),
                'unit': 'V',
                'description': 'Offset for X axis'
            },
            'offset_y': {
                'type': 'float',
                'default': 0.0,
                'range': (-10.0, 10.0),
                'unit': 'V',
                'description': 'Offset for Y axis'
            },
            'x_pixels': {
                'type': 'int',
                'default': 512,
                'range': (64, 4096),
                'description': 'Number of X pixels'
            },
            'y_pixels': {
                'type': 'int',
                'default': 512,
                'range': (64, 4096),
                'description': 'Number of Y pixels'
            },
            'numtiles_x': {
                'type': 'int',
                'default': 10,
                'range': (1, 1000),
                'description': 'Number of X tiles'
            },
            'numtiles_y': {
                'type': 'int',
                'default': 10,
                'range': (1, 1000),
                'description': 'Number of Y tiles'
            },
            'numtiles_z': {
                'type': 'int',
                'default': 5,
                'range': (1, 1000),
                'description': 'Number of Z tiles'
            },
            'tile_size_x': {
                'type': 'float',
                'default': 100.0,
                'range': (-10000, 10000),
                'unit': 'μm',
                'description': 'X tile size'
            },
            'tile_size_y': {
                'type': 'float',
                'default': 100.0,
                'range': (-10000, 10000),
                'unit': 'μm',
                'description': 'Y tile size'
            },
            'tile_size_z': {
                'type': 'float',
                'default': 50.0,
                'range': (-10000, 10000),
                'unit': 'μm',
                'description': 'Z tile size'
            }
        }
    
    @property
    def acquisition_class(self) -> Type:
        return ConfocalMosaic


class ConfocalMosaic(Acquisition):
    def __init__(self, galvo=None, data_inputs=None, prior_stage=None, num_frames=1, signal_bus=None, acquisition_parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.galvo = galvo
        self.data_inputs = data_inputs or []
        self.prior_stage = prior_stage
        self.num_frames = num_frames
        self.signal_bus = signal_bus
        self.verified = False
        self.acquisition_parameters = acquisition_parameters or {}
        
        if self.galvo is None:
            raise ValueError("Galvo is required for ConfocalMosaic acquisition")
        if not self.data_inputs:
            raise ValueError("At least one DataInput is required for ConfocalMosaic acquisition")
        if self.prior_stage is None:
            raise ValueError("Prior stage is required for ConfocalMosaic acquisition")
        
        self.rpoc_enabled = False
        self.rpoc_mask_channels = {}
        self.rpoc_static_channels = {}
        self.rpoc_script_channels = {}
        self.rpoc_ttl_signals = {}  # channel_id -> flat TTL array
        
        # Initialize static channels list - will be populated when static channels are configured
        self._static_channels = []

    def configure_rpoc(self, rpoc_enabled, rpoc_mask_channels=None, rpoc_static_channels=None, rpoc_script_channels=None, **kwargs):
        self.rpoc_enabled = rpoc_enabled
        self.rpoc_mask_channels = rpoc_mask_channels or {}
        self.rpoc_static_channels = rpoc_static_channels or {}
        self.rpoc_script_channels = rpoc_script_channels or {}
        self.rpoc_ttl_signals = {}  # channel_id -> flat TTL array

        # Clear existing static channels list
        self._static_channels = []

        # Prepare static channels list for on-demand task creation
        if self.rpoc_static_channels:
            device_name = self.galvo.parameters.get('device_name', 'Dev1') if self.galvo else 'Dev1'
            
            for channel_id, channel_data in self.rpoc_static_channels.items():
                device = channel_data.get('device', device_name)
                channel_id_int = int(channel_id) if isinstance(channel_id, str) else channel_id
                port_line = channel_data.get('port_line', f'port1/line{channel_id_int-1}')
                channel_name = f"{device}/{port_line}"
                self._static_channels.append(channel_name)

        # Get acquisition parameters
        dwell_time = self.acquisition_parameters.get('dwell_time', 10)  # microseconds
        rate = self.galvo.parameters.get('sample_rate', 1000000)
        dwell_time_sec = dwell_time / 1e6
        pixel_samples = max(1, int(dwell_time_sec * rate))
        numsteps_x = self.acquisition_parameters.get('x_pixels', 512)
        numsteps_y = self.acquisition_parameters.get('y_pixels', 512)
        extra_left = self.acquisition_parameters.get('extrasteps_left', 50)
        extra_right = self.acquisition_parameters.get('extrasteps_right', 50)
        total_x = numsteps_x + extra_left + extra_right
        total_y = numsteps_y

        # Handle mask-based channels
        for channel_id, channel_data in (self.rpoc_mask_channels or {}).items():
            mask = channel_data.get('mask_data')
            if mask is None:
                continue
            # Prepare mask
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
            # Pad mask to match scan size
            padded_mask = []
            for row in range(numsteps_y):
                padded_row = np.concatenate((
                    np.zeros(extra_left, dtype=bool),
                    mask_array[row, :] if row < mask_array.shape[0] else np.zeros(numsteps_x, dtype=bool),
                    np.zeros(extra_right, dtype=bool)
                ))
                padded_mask.append(padded_row)
            padded_mask = np.array(padded_mask)
            # For each pixel, create TTL: high for all samples if mask is high, else all low
            ttl = np.zeros((total_y, total_x, pixel_samples), dtype=bool)
            for y in range(total_y):
                for x in range(total_x):
                    if padded_mask[y, x]:
                        ttl[y, x, :] = True
            flat_ttl = ttl.ravel()
            self.rpoc_ttl_signals[channel_id] = flat_ttl

        # Handle static channels
        for channel_id, channel_data in (self.rpoc_static_channels or {}).items():
            level = channel_data.get('level', 'Static Low').lower()
            # All high or all low
            value = True if 'high' in level else False
            flat_ttl = np.full(total_x * total_y * pixel_samples, value, dtype=bool)
            self.rpoc_ttl_signals[channel_id] = flat_ttl

        if self.signal_bus:
            n_masks = len(self.rpoc_mask_channels) if self.rpoc_mask_channels else 0
            n_static = len(self.rpoc_static_channels) if self.rpoc_static_channels else 0
            n_script = len(self.rpoc_script_channels) if self.rpoc_script_channels else 0
            n_total = len(self.rpoc_ttl_signals)

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

        numtiles_x = int(self.acquisition_parameters.get('numtiles_x', 1))
        numtiles_y = int(self.acquisition_parameters.get('numtiles_y', 1))
        numtiles_z = int(self.acquisition_parameters.get('numtiles_z', 1))
        tile_size_x = int(self.acquisition_parameters.get('tile_size_x', 100))  # in microns
        tile_size_y = int(self.acquisition_parameters.get('tile_size_y', 100))  # in microns
        tile_size_z = float(self.acquisition_parameters.get('tile_size_z', 50))  # in microns
        
        try:
            start_x, start_y = self.prior_stage.get_xy()  # in microns (int)
            start_z = self.prior_stage.get_z()  # in 0.1 micron units (int)
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error getting current stage position: {e}")
            raise RuntimeError(f"Failed to get current stage position: {e}")
        
        stage_step_sizes = []
        tile_indices = []
        for z_idx in range(numtiles_z):
            for y_idx in range(numtiles_y):
                for x_idx in range(numtiles_x):
                    # Calculate step sizes relative to current position
                    # For the first position (0,0,0), step is 0 (stay at current position)
                    # For subsequent positions, step is the difference from the previous position
                    if x_idx == 0 and y_idx == 0 and z_idx == 0:
                        # First position - stay at current position
                        x_step = 0
                        y_step = 0
                        z_step = 0
                    else:
                        # Calculate step based on the grid traversal pattern
                        # Grid is traversed: (0,0,0) -> (1,0,0) -> (2,0,0) -> ... -> (0,1,0) -> (1,1,0) -> ... -> (0,0,1) -> ...
                        if x_idx > 0:
                            # Moving in X direction
                            x_step = tile_size_x
                            y_step = 0
                            z_step = 0
                        elif y_idx > 0:
                            # Moving in Y direction (reset X)
                            x_step = -x_idx * tile_size_x  # Reset X to 0
                            y_step = tile_size_y
                            z_step = 0
                        elif z_idx > 0:
                            # Moving in Z direction (reset X and Y)
                            x_step = -x_idx * tile_size_x  # Reset X to 0
                            y_step = -y_idx * tile_size_y  # Reset Y to 0
                            z_step = int(tile_size_z * 10)
                        else:
                            x_step = 0
                            y_step = 0
                            z_step = 0
                    stage_step_sizes.append((x_step, y_step, z_step))
                    tile_indices.append((x_idx, y_idx, z_idx))

        all_frames = []
        total_positions = len(stage_step_sizes) * self.num_frames
        current_position = 0
        
        
        
        for frame_idx in range(self.num_frames):
            if self._stop_flag and self._stop_flag():
                break
            
            for pos_idx, (x_step, y_step, z_step) in enumerate(stage_step_sizes):
                if self._stop_flag and self._stop_flag():
                    break
                
                try:
                    # Get current position and calculate target position
                    current_x, current_y = self.prior_stage.get_xy()
                    current_z = self.prior_stage.get_z()
                    
                    target_x = int(current_x + x_step)
                    target_y = int(current_y + y_step)
                    target_z = int(current_z + z_step)
                    
                    if self.signal_bus:
                        self.signal_bus.console_message.emit(f"Moving to position {pos_idx + 1}/{len(stage_step_sizes)}: X={target_x} µm, Y={target_y} µm, Z={target_z/10:.1f} µm (step: +{x_step}, +{y_step}, +{z_step/10:.1f})")
                    self.prior_stage.move_xy(target_x, target_y)
                    self.prior_stage.move_z(target_z)
                    time.sleep(0.5)
                except Exception as e:
                    if self.signal_bus:
                        self.signal_bus.console_message.emit(f"Error moving stage: {e}")
                    raise RuntimeError(f"Failed to move stage to position {pos_idx + 1}: {e}")

                frame_data = self.collect_confocal_data(self.galvo, ai_channels)
                
                if self.signal_bus:
                    # Use new uniform pipeline instead of legacy data_signal
                    self.emit_data(self.signal_bus, frame_data)
                all_frames.append(frame_data)
                current_position += 1
            
            # Return stage to original position after completing all tiles for this frame
            # (but only if there are more frames to process)
            if frame_idx < self.num_frames - 1:
                try:
                    
                    move_z = ((numtiles_z) - 1) * tile_size_z * 10
                    print(f'move_z: {move_z}')
                    move_y = ((numtiles_y) - 1) * tile_size_y
                    move_x = ((numtiles_x) - 1) * tile_size_x
                    
                    curr_z = self.prior_stage.get_z()
                    print(curr_z)
                    curr_x, curr_y = self.prior_stage.get_xy()

                    if self.signal_bus:
                        self.signal_bus.console_message.emit(f"Frame {frame_idx + 1} complete. Returning stage to original position for next frame: X={curr_x - move_x} µm, Y={curr_y - move_y} µm, Z={(curr_z - move_z)/10:.1f} µm")

                    self.prior_stage.move_xy(curr_x - move_x, curr_y - move_y)
                    self.prior_stage.move_z(curr_z - move_z)

                    time.sleep(0.5)
                    if self.signal_bus:
                        self.signal_bus.console_message.emit("Stage returned to original position successfully")
                except Exception as e:
                    if self.signal_bus:
                        self.signal_bus.console_message.emit(f"Warning: Failed to return stage to original position after frame {frame_idx + 1}: {e}")
        
        if all_frames:
            final_data = np.stack(all_frames)
            if self.signal_bus:
                # Use new uniform pipeline instead of legacy data_signal
                self.emit_acquisition_complete(self.signal_bus)
            
            self.metadata['tile_order'] = tile_indices            
            self.save_data(final_data)
            
            # Return stage to original position (final return)
            try:
                

                move_z = ((numtiles_z) - 1) * tile_size_z * 10
                print(f'move_z: {move_z}')
                move_y = ((numtiles_y) - 1) * tile_size_y
                move_x = ((numtiles_x) - 1) * tile_size_x
                
                curr_z = self.prior_stage.get_z()
                print(curr_z)
                curr_x, curr_y = self.prior_stage.get_xy()

                if self.signal_bus:
                    self.signal_bus.console_message.emit(f"All frames complete. Returning stage to original position: X={curr_x - move_x} µm, Y={curr_y - move_y} µm, Z={(curr_z - move_z)/10:.1f} µm")

                self.prior_stage.move_xy(curr_x - move_x, curr_y - move_y)
                self.prior_stage.move_z(curr_z - move_z)
                time.sleep(0.5)
                if self.signal_bus:
                    self.signal_bus.console_message.emit("Stage returned to original position successfully")
            except Exception as e:
                if self.signal_bus:
                    self.signal_bus.console_message.emit(f"Warning: Failed to return stage to original position: {e}")
            
            return final_data
        else:
            # Return stage to original position even if no frames were acquired
            try:
                if self.signal_bus:
                    self.signal_bus.console_message.emit(f"Returning stage to original position: X={start_x} µm, Y={start_y} µm, Z={start_z/10:.1f} µm")
                self.prior_stage.move_xy(start_x, start_y)
                self.prior_stage.move_z(start_z)
                time.sleep(0.5)
                if self.signal_bus:
                    self.signal_bus.console_message.emit("Stage returned to original position successfully")
            except Exception as e:
                if self.signal_bus:
                    self.signal_bus.console_message.emit(f"Warning: Failed to return stage to original position: {e}")
            
            return None
    
    def collect_confocal_data(self, galvo, ai_channels):
        try:
            rate = galvo.parameters.get('sample_rate', 1000000)
            dwell_time = self.acquisition_parameters.get('dwell_time', 10)  # microseconds
            dwell_time_sec = dwell_time / 1e6  # convert to seconds
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

            rpoc_do_channels = []
            rpoc_ttl_signals = []

            if self.rpoc_enabled and self.rpoc_ttl_signals and (self.rpoc_mask_channels or self.rpoc_static_channels or self.rpoc_script_channels):
                for channel_id, flat_ttl in self.rpoc_ttl_signals.items():
                    # Find the channel info from the appropriate storage
                    channel_info = None
                    if channel_id in self.rpoc_mask_channels:
                        channel_info = self.rpoc_mask_channels[channel_id]
                    elif channel_id in self.rpoc_static_channels:
                        channel_info = self.rpoc_static_channels[channel_id]
                    elif channel_id in self.rpoc_script_channels:
                        channel_info = self.rpoc_script_channels[channel_id]
                    
                    if channel_info:
                        device = channel_info.get('device', device_name)
                        # Convert channel_id to int for arithmetic if it's a string
                        channel_id_int = int(channel_id) if isinstance(channel_id, str) else channel_id
                        port_line = channel_info.get('port_line', f'port0/line{4+channel_id_int-1}')
                        rpoc_do_channels.append(f"{device}/{port_line}")
                        rpoc_ttl_signals.append(flat_ttl)

            # Calculate timeout once
            timeout = total_samples / rate + 5
            
            # Separate dynamic (port0) and static (port1+) channels
            dyn_chans = []
            dyn_ttls = []
            stat_vals = []
            stat_chans = []
            
            for chan, flat_ttl in zip(rpoc_do_channels, rpoc_ttl_signals):
                if '/port0/' in chan.lower():
                    # anything on port0 → dynamic, clocked DO
                    dyn_chans.append(chan)
                    dyn_ttls.append(flat_ttl)
                else:
                    # anything else (e.g. port1) → static DO
                    # take the first value as constant level
                    stat_vals.append(bool(flat_ttl.flat[0]))
                    stat_chans.append(chan)

            # -- static, immediate DO task (on port1 or others) --
            if stat_vals and stat_chans:
                # 1) Raise static lines before imaging:
                self.write_static(stat_vals, stat_chans)

            try:
                # -- dynamic, hardware-timed DO task (on port0) --
                if dyn_chans:
                    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
                        # 1) Add AO & AI channels
                        ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_channel}")
                        ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_channel}")
                        for ch in ai_channels:
                            ai_task.ai_channels.add_ai_voltage_chan(ch)
                        
                        # 2) Clock AO
                        ao_task.timing.cfg_samp_clk_timing(
                            rate=rate,
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=total_samples
                        )
                        
                        # 3) Clock AI off of AO's internal clock
                        ai_task.timing.cfg_samp_clk_timing(
                            rate=rate,
                            source=f"/{device_name}/ao/SampleClock",
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=total_samples
                        )
                        
                        # 4) Clock DO off of AO's internal clock (AO still open!)
                        for c in dyn_chans:
                            do_task.do_channels.add_do_chan(c)
                        do_task.timing.cfg_samp_clk_timing(
                            rate=rate,
                            source=f"/{device_name}/ao/SampleClock",
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=total_samples
                        )
                        
                        # 5) Write waveforms, then start in order:
                        ao_task.write(waveform, auto_start=False)
                        
                        # write the pattern(s)
                        if len(dyn_chans) == 1:
                            do_task.write(dyn_ttls[0].tolist(), auto_start=False)
                        else:
                            data_to_write = [arr.tolist() for arr in dyn_ttls]
                            do_task.write(data_to_write, auto_start=False)
                        
                        # 6) Start in order: AI, AO, DO
                        ai_task.start()
                        do_task.start()
                        ao_task.start()
                        
                        
                        # 7) Wait and tear down all three
                        ao_task.wait_until_done(timeout=timeout)
                        ai_task.wait_until_done(timeout=timeout)
                        do_task.wait_until_done(timeout=timeout)
                        
                        acq_data = np.array(ai_task.read(number_of_samples_per_channel=total_samples))
                else:
                    # No dynamic DO channels, just AO and AI
                    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
                        # 1) Add AO & AI channels
                        ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{fast_channel}")
                        ao_task.ao_channels.add_ao_voltage_chan(f"{device_name}/ao{slow_channel}")
                        for ch in ai_channels:
                            ai_task.ai_channels.add_ai_voltage_chan(ch)
                        
                        # 2) Clock AO
                        ao_task.timing.cfg_samp_clk_timing(
                            rate=rate,
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=total_samples
                        )
                        
                        # 3) Clock AI off of AO's internal clock
                        ai_task.timing.cfg_samp_clk_timing(
                            rate=rate,
                            source=f"/{device_name}/ao/SampleClock",
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=total_samples
                        )
                        
                        # 4) Write waveforms, then start in order:
                        ao_task.write(waveform, auto_start=False)
                        
                        # 5) Start in order: AI, AO
                        ai_task.start()
                        ao_task.start()
                        
                        # 6) Wait and tear down
                        ao_task.wait_until_done(timeout=timeout)
                        ai_task.wait_until_done(timeout=timeout)
                        
                        acq_data = np.array(ai_task.read(number_of_samples_per_channel=total_samples))
            finally:
                # Always clear static lines after, even if an exception occurred
                if stat_vals and stat_chans:
                    try:
                        self.write_static([False] * len(stat_vals), stat_chans)
                    except:
                        pass

            input_results = []
            
            for i in range(len(ai_channels)):
                channel_data = acq_data if len(ai_channels) == 1 else acq_data[i]
                reshaped = channel_data.reshape(total_y, total_x, pixel_samples)
                
                # Average over all samples for each pixel (no splitting)
                averaged_data = np.mean(reshaped, axis=2)
                
                # Crop to remove extra steps
                cropped_data = averaged_data[:, extra_left:extra_left + numsteps_x]
                input_results.append(cropped_data)
            
            return np.stack(input_results)
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
                num_total_images, num_channels, height, width = data.shape
            elif len(data.shape) == 3:
                num_total_images, height, width = data.shape
                num_channels = 1
                data = data.reshape(num_total_images, 1, height, width)
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
            
            input_channel_names = self._get_input_channel_names()
            
            for input_ch_idx in range(num_channels):
                input_channel_name = input_channel_names.get(input_ch_idx, f"input{input_ch_idx}")
                
                channel_data = data[:, input_ch_idx, :, :]
                filename = f"{Path(self.save_path).stem}_{input_channel_name}.tiff"
                filepath = save_dir / filename
                tifffile.imwrite(filepath, channel_data)
                
                if self.signal_bus:
                    self.signal_bus.console_message.emit(f"Saved confocal mosaic data for {input_channel_name}")
            
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error saving confocal mosaic data: {e}")
            else:
                print(f"Error saving confocal mosaic data: {e}")
    
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
    
    def write_static(self, levels, channel_names=None):
        """Write a list of boolean levels to the static channels using on-demand task creation."""
        if not levels:
            # No levels to write
            return
        
        if not channel_names:
            # If no channel names provided, use all static channels (backward compatibility)
            channel_names = self._static_channels[:len(levels)]
        
        if len(levels) != len(channel_names):
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Warning: Number of levels ({len(levels)}) doesn't match number of channels ({len(channel_names)})")
            return
        
        # Create a new task for each write operation to avoid resource conflicts
        try:
            with nidaqmx.Task() as static_task:
                for channel_name in channel_names:
                    static_task.do_channels.add_do_chan(channel_name)
                # Write the levels immediately
                static_task.write(levels, auto_start=True)
        except Exception as e:
            if self.signal_bus:
                self.signal_bus.console_message.emit(f"Error writing to static channels: {e}")

    def cleanup(self):
        """Clean up resources."""
        self._static_channels = []

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
    
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
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                frame[ch, mask] += intensity

            frame[ch] += np.random.normal(0, 0.1, frame[ch].shape)
            frame[ch] = np.clip(frame[ch], 0, 1)

        if self.signal_bus:
            n_masks = len(self.rpoc_mask_channels) if self.rpoc_mask_channels else 0
            n_static = len(self.rpoc_static_channels) if self.rpoc_static_channels else 0
            n_script = len(self.rpoc_script_channels) if self.rpoc_script_channels else 0
            self.signal_bus.console_message.emit(f"RPOC Debug - enabled: {self.rpoc_enabled}, masks: {n_masks}, static: {n_static}, script: {n_script}")
        
        if self.rpoc_enabled and (self.rpoc_mask_channels or self.rpoc_static_channels):
            # Handle mask channels
            if self.rpoc_mask_channels:
                combined_mask = np.zeros((y_pixels, x_pixels), dtype=bool)
                
                for channel_id, channel_data in self.rpoc_mask_channels.items():
                    mask = channel_data.get('mask_data')
                    if mask is None:
                        continue
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
                    if mask_array.shape != (y_pixels, x_pixels):
                        from scipy.ndimage import zoom
                        zoom_factors = (y_pixels / mask_array.shape[0], x_pixels / mask_array.shape[1])
                        mask_array = zoom(mask_array, zoom_factors, order=0).astype(bool)
                    
                    combined_mask |= mask_array
                
                for ch in range(num_channels):
                    frame[ch, combined_mask] = 0
            
            # Handle static channels (simulation only shows they're configured)
            if self.rpoc_static_channels:
                for ch in range(num_channels):
                    if np.max(frame[ch]) > 0:
                        frame[ch] = frame[ch] / np.max(frame[ch])
            
            if self.signal_bus:
                rpoc_channels_info = []
                for channel_id, channel_data in self.rpoc_mask_channels.items():
                    device = channel_data.get('device', 'Dev1')
                    port_line = channel_data.get('port_line', f'port0/line{4+channel_id-1}')
                    rpoc_channels_info.append(f"Channel {channel_id}: {device}/{port_line}")
                
                static_channels_info = []
                for channel_id, channel_data in self.rpoc_static_channels.items():
                    device = channel_data.get('device', 'Dev1')
                    port_line = channel_data.get('port_line', f'port0/line{4+channel_id-1}')
                    level = channel_data.get('level', 'Static Low')
                    static_channels_info.append(f"Channel {channel_id}: {device}/{port_line} ({level})")
                
                total_info = rpoc_channels_info + static_channels_info
                self.signal_bus.console_message.emit(f"RPOC Active - {len(self.rpoc_mask_channels)} masks, {len(self.rpoc_static_channels)} static channels: {', '.join(total_info)}")

        time.sleep(1)
        
        return frame