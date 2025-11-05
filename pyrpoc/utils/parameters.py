from dataclasses import dataclass
import numpy as np
from typing import Any, Optional
from abc import ABC, abstractmethod


class BaseParameter(ABC):
    '''
    description: 
        parameter object that has all the relevant attributes for
        a modality and for the GUI to read

    args:
        name: parameter identifier
        value: parameter value
        group: (optional) parameters with same group name are put together on GUI side
        range: (optional) valid range for the parameter on GUI side
        default: (optional) value loaded into GUI initially when displayed
    '''
    name: str
    value: Any
    default: Optional[Any] = None
    group: Optional[str] = None
    range: Optional[tuple] = None
    

    def create_widget(self):
        pass

    def get_value(self):
        pass

    def get_widget(self):
        pass

class SavePath(BaseParameter):
    ...

class SaveEnable(BaseParameter):
    ...

class NumericalValue(BaseParameter):
    ...

class TextValue(BaseParameter):
    ...