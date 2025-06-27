import numpy as np
import nidaqmx
import abc
from pyrpoc.imaging.instruments import *
import time

class Acquisition(abc.ABC):
    def __init__(self):
        self._stop_flag = None

    def set_stop_flag(self, stop_flag_func):
        '''
        stop button in main gui for in any given acquisition sets this flag
        '''
        self._stop_flag = stop_flag_func

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

    
class RPOCHandler():
    def __init__(self):
        self.script_chans = {'port0/line4': 'line4.py', 'port0/line5': 'line5.py'}
        self.static_chans = {'port0/line6': 'line4.py', 'port0/line5': 'line5.py'}


'''
things to check in each class
1. needs the methods, abc takes cares of that
2. have a flag, self.verified, that confirms that all instruments are safely connected when the first acquisition is done
    if self.verified is verified, then dont perform the check subsequently
'''

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
                self.signal_bus.frame_acquired.emit(frame_data, frame, self.num_frames)
            time.sleep(1)
        
        if frames:
            return np.stack(frames)
        else:
            return None
    





class Confocal(Acquisition):
    def __init__(self, galvo: Galvo, input_chans: list[str]):
        pass
        

class Widefield(Acquisition):
    def __init__(self):
        pass

class Hyperspectral(Acquisition):
    def __init__(self):
        pass

class ZScan(Acquisition):
    def __init__(self):
        pass

class Mosaic(Acquisition):
    def __init__(self):
        pass

class Custom(Acquisition):
    def __init__(self):
        pass