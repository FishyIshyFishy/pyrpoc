import numpy as np
import nidaqmx
import abc
from pyrpoc.imaging.instruments import *

class Acquisition(abc.ABC):
    def __init__(self):
        self.acquisition_type = 'something'
        self.rpoc_something = 'something'
        self.verified = False
        pass

    @abc.abstractmethod
    def verify_acquisition(self):
        '''
        check self.verified, and make sure all instruments are safely connected if not yet verified
        '''
        self.verified = True
        return self.verified
    
    @abc.abstractmethod
    def configure_rpoc(self, rpoc_enabled, **kwargs):
        '''
        check global rpoc flag (whatever that ends up being)
        if not, set up rpoc for the acquisition process
        '''

    @abc.abstractmethod
    def configure_imaging(self, imaging_mode):
        '''
        set up widefield or confocal as needed
        '''

    @abc.abstractmethod
    def perform_acquisition(self):
        '''
        this will be the one that varies a lot between acquisition types
        '''
        pass    

    @abc.abstractmethod
    def finalize_acquisition(self):
        '''
        prep the data output for whatever format the GUI ends up wanting
        '''
        data = None
        return data
    
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
    def __init__(self, input_chans: list[str]):
        super().__init__(self)
        pass

    def perform_acquisition(self):
        return super().perform_acquisition()
    





class Confocal(Acquisition):
    def __init__(self, galvo: Galvo, input_chans: list[str]):
        super().__init__(self)
        pass
        

class Widefield(Acquisition):
    def __init__(self):
        super().__init__(self)
        pass

class Hyperspectral(Acquisition):
    def __init__(self):
        super().__init__(self)
        pass

class ZScan(Acquisition):
    def __init__(self):
        super().__init__(self)
        pass

class Mosaic(Acquisition):
    def __init__(self):
        super().__init__(self)
        pass

class Custom(Acquisition):
    def __init__(self):
        super().__init__(self)
        pass