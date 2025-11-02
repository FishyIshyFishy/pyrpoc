from dataclasses import dataclass
import numpy as np
from typing import Any, Optional

@dataclass
class BaseParameter:
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
    