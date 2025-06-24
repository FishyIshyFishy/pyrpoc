import numpy as np
import nidaqmx
import abc
from pyrpoc.imaging.instruments import *
import time

class Acquisition(abc.ABC):
    def __init__(self):
        pass

    
    @abc.abstractmethod
    def configure_rpoc(self, rpoc_enabled, **kwargs):
        '''
        check global rpoc flag (whatever that ends up being)
        if not, set up rpoc for the acquisition process
        '''

    @abc.abstractmethod
    def perform_acquisition(self): 
        '''
        this will be the one that varies a lot between acquisition types
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
        pass

    def configure_rpoc(self, rpoc_enabled, **kwargs):
        pass

    def perform_acquisition(self):
        self.data = np.zeros((self.num_frames, self.y_pixels, self.x_pixels))
        for frame in range(self.num_frames):
            frame_data = np.zeros((self.y_pixels, self.x_pixels))
            for y in range(self.y_pixels):
                for x in range(self.x_pixels):
                    frame_data[y, x] = np.random.rand()
            self.data[frame] = frame_data
            if self.signal_bus:
                self.signal_bus.frame_acquired.emit(frame_data, frame, self.num_frames)
            time.sleep(1)
            yield frame_data
        # After all frames, return the full stack
        return self.data
    





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