import numpy as np
import abc

class Instrument(abc.ABC):
    def __init__(self, name, io_type, chan_type, chan):
        self.name = name
        self.instr_io_type = io_type # input, output
        self.instr_chan_type = chan_type # analog_out, digital_out, analog_in, digital_in, com
        self.instr_chan = chan 

    @abc.abstractmethod
    def initialize(self):
        pass

    def get_instrument_info(self):
        return f'<{self.__class__.__name__}: {self.name} on {self.instr_chan}. \n io_type: {self.instr_io_type} \n instr_chan_type: {self.instr_chan_type}'

class Galvo(Instrument):
    def __init__(self):
        super().__init__()
        pass  

class AOM(Instrument):
    def __init__(self):
        super().__init__()
        pass  

class ZaberStage(Instrument):
    def __init__(self):
        super().__init__()
        pass  