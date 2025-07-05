import numpy as np
import nidaqmx
import abc
from pyrpoc.imaging.instruments import *
import time
import nidaqmx
from nidaqmx.constants import AcquisitionType

class Acquisition(abc.ABC):
    def __init__(self):
        self._stop_flag = None

    def set_stop_flag(self, stop_flag_func):
        '''
        stop button in main gui for in any given acquisition sets this flag
        '''
        self._stop_flag = stop_flag_func
    
    def set_worker(self, worker):
        '''
        Set reference to worker for signal emission
        '''
        self.worker = worker

    @abc.abstractmethod
    def configure_rpoc(self, rpoc_enabled, **kwargs):
        '''
        check global rpoc flag (whatever that ends up being)
        if not, set up rpoc for the acquisition process
        '''

    @abc.abstractmethod
    def perform_acquisition(self): 
        '''
        yield each lowest-level data unit (e.g., a single image, a single tile, etc.) as it is acquired, and finally return a list or array of all such data units
        '''
        pass    



class Simulated(Acquisition):
    def __init__(self, x_pixels: int, y_pixels: int, num_frames: int, signal_bus=None):
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.num_frames = num_frames
        self.signal_bus = signal_bus

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        frames = []
        for frame in range(self.num_frames):
            # Check if we should stop
            if self._stop_flag and self._stop_flag():
                break
                
            frame_data = np.random.rand(self.y_pixels, self.x_pixels)
            frames.append(frame_data)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame, self.num_frames, False)
            time.sleep(1)
        
        if frames:
            final_data = np.stack(frames)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, len(frames)-1, self.num_frames, True)
            return final_data
        else:
            return None
    





class Confocal(Acquisition):
    def __init__(self, galvo=None, data_inputs=None, num_frames=1, signal_bus=None, **kwargs):
        super().__init__()
        self.galvo = galvo
        self.data_inputs = data_inputs or []
        self.num_frames = num_frames
        self.signal_bus = signal_bus
        self.verified = False
        
        # Acquisition parameters
        self.rpoc_enabled = False
        self.rpoc_masks = {}
        
        # Store reference to worker for signal emission
        self.worker = None

    def configure_rpoc(self, rpoc_enabled, rpoc_masks=None, **kwargs):
        self.rpoc_enabled = rpoc_enabled
        if rpoc_masks:
            self.rpoc_masks = rpoc_masks

    def perform_acquisition(self):     
        # Get pixel dimensions from galvo parameters
        x_pixels = self.galvo.parameters.get('numsteps_x', 512)
        y_pixels = self.galvo.parameters.get('numsteps_y', 512)
        
        # Generate galvo waveform
        try:
            waveform = self.galvo.generate_raster_waveform(x_pixels, y_pixels)
        except Exception as e:
            print(f"Error generating galvo waveform: {e}")
            return None
        
        # Get data input channels
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
            
            frame_data = self.generate_simulated_confocal()
            # frame_data = self.collect_data(galvo, ai_channels)
            
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame_idx, self.num_frames, False)
            all_frames.append(frame_data)
        
        if all_frames:
            final_data = np.stack(all_frames)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, len(all_frames)-1, self.num_frames, True)
            return final_data
        else:
            return None
    
    def generate_simulated_confocal(self):
        # Get pixel dimensions from galvo parameters
        x_pixels = self.galvo.parameters.get('numsteps_x', 512)
        y_pixels = self.galvo.parameters.get('numsteps_y', 512)
        
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

        time.sleep(1)
        
        return frame
    
    def collect_data(self, galvo, ai_channels):
        try:
            rate = galvo.parameters.get('sample_rate', 1000000)
            dwell_time = galvo.parameters.get('dwell_time', 10e-6)
            extra_left = galvo.parameters.get('extrasteps_left', 50)
            extra_right = galvo.parameters.get('extrasteps_right', 50)
            numsteps_x = galvo.parameters.get('numsteps_x', 512)
            numsteps_y = galvo.parameters.get('numsteps_y', 512)
            slow_channel = galvo.parameters.get('slow_axis_channel', 0)
            fast_channel = galvo.parameters.get('fast_axis_channel', 1)
            device_name = galvo.parameters.get('device_name', 'Dev1')

            pixel_samples = max(1, int(dwell_time * rate))
            total_x = numsteps_x + extra_left + extra_right
            total_y = numsteps_y
            total_samples = total_x * total_y * pixel_samples

            waveform = galvo.generate_raster_waveform(numsteps_x, numsteps_y)
            
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

                ao_task.write(waveform.T, auto_start=False)  # transpose to match DAQ format
                ai_task.start()
                ao_task.start()

                timeout = total_samples / rate + 5
                ao_task.wait_until_done(timeout=timeout)
                ai_task.wait_until_done(timeout=timeout)
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
                else:
                    return np.stack(results)
                    
        except Exception as e:
            print(f"Error in DAQ acquisition: {e}")
            return self.generate_simulated_confocal()

class Widefield(Acquisition):
    def __init__(self, data_inputs=None, signal_bus=None, **kwargs):
        super().__init__()
        self.data_inputs = data_inputs or []
        self.signal_bus = signal_bus
        self.verified = False

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        # TODO: Implement widefield acquisition
        # For now, return simulated data with frame emission
        frames = []
        for frame in range(1):  # Single frame for now
            if self._stop_flag and self._stop_flag():
                break
            frame_data = np.random.rand(512, 512)
            frames.append(frame_data)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame, 1, False)
        
        if frames:
            final_data = frames[0]  # Return single frame
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, 0, 1, True)
            return final_data
        else:
            return None

class Hyperspectral(Acquisition):
    def __init__(self, signal_bus=None, **kwargs):
        super().__init__()
        self.signal_bus = signal_bus
        self.verified = False

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        # TODO: Implement hyperspectral acquisition
        # For now, return simulated data with frame emission
        frames = []
        for frame in range(1):  # Single frame for now
            if self._stop_flag and self._stop_flag():
                break
            frame_data = np.random.rand(512, 512)
            frames.append(frame_data)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame, 1, False)
        
        if frames:
            final_data = frames[0]  # Return single frame
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, 0, 1, True)
            return final_data
        else:
            return None

class ZScan(Acquisition):
    def __init__(self, stages=None, signal_bus=None, **kwargs):
        super().__init__()
        self.stages = stages or []
        self.signal_bus = signal_bus
        self.verified = False

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        # TODO: Implement ZScan acquisition
        # For now, return simulated data with frame emission
        frames = []
        for frame in range(1):  # Single frame for now
            if self._stop_flag and self._stop_flag():
                break
            frame_data = np.random.rand(512, 512)
            frames.append(frame_data)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame, 1, False)
        
        if frames:
            final_data = frames[0]  # Return single frame
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, 0, 1, True)
            return final_data
        else:
            return None

class Mosaic(Acquisition):
    def __init__(self, stages=None, signal_bus=None, **kwargs):
        super().__init__()
        self.stages = stages or []
        self.signal_bus = signal_bus
        self.verified = False

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        # TODO: Implement mosaic acquisition
        # For now, return simulated data with frame emission
        frames = []
        for frame in range(1):  # Single frame for now
            if self._stop_flag and self._stop_flag():
                break
            frame_data = np.random.rand(512, 512)
            frames.append(frame_data)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame, 1, False)
        
        if frames:
            final_data = frames[0]  # Return single frame
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, 0, 1, True)
            return final_data
        else:
            return None

class Custom(Acquisition):
    def __init__(self, signal_bus=None, **kwargs):
        super().__init__()
        self.signal_bus = signal_bus
        self.verified = False

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        # TODO: Implement custom acquisition
        # For now, return simulated data with frame emission
        frames = []
        for frame in range(1):  # Single frame for now
            if self._stop_flag and self._stop_flag():
                break
            frame_data = np.random.rand(512, 512)
            frames.append(frame_data)
            if self.signal_bus:
                self.signal_bus.data_signal.emit(frame_data, frame, 1, False)
        
        if frames:
            final_data = frames[0]  # Return single frame
            if self.signal_bus:
                self.signal_bus.data_signal.emit(final_data, 0, 1, True)
            return final_data
        else:
            return None