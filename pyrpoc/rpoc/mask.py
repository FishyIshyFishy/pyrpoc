import numpy as np
import abc

class Mask:
    def __init__(self):
        pass

    @abc.abstractmethod
    def create_mask(self):
        pass

class RPOC_Script(Mask):
    def __init__(self):
        super().__init__()
        pass

class RPOC_Image(Mask):
    def __init__(self):
        super().__init__()
        pass