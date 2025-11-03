from dataclasses import dataclass
import numpy as np
from typing import Any


@dataclass
class BaseData:
    '''
    description:
        base class for all data types emitted by acquisitions and consumed
        by displays. Holds a name, a numeric value or array, and optional metadata.

    args:
        name: identifier for the data (e.g., 'channel_0')
        value: the data itself
        ndim: number of dimensions of the data (0 = scalar, 1 = row, 2 = image, 3 = volume).
        metadata: optional dictionary of extra information (units, voxel sizes, axis labels, whatever)
            ***UNDER NO CIRCUMSTANCES SHOULD METADATA BE ACCESSED BY DISPLAY CODE***

    example:
        base = BaseData(name='raw', value=np.array([1, 2, 3]), ndim=1)
    '''
    name: str
    value: np.ndarray | float
    ndim: int
    metadata: dict[str, Any] | None = None


@dataclass
class DataScalar(BaseData):
    '''
    description:
        a single scalar value of data, ndim fixed at 0.

    args:
        name: identifier for the data (e.g., 'channel_0')
        value: the data itself (float value)
        metadata: optional dictionary of extra information (units, voxel sizes, axis labels, whatever)
            ***UNDER NO CIRCUMSTANCES SHOULD METADATA BE ACCESSED BY DISPLAY CODE***

    example:
        scalar = DataScalar(name='chan_0', value=42.0)
    '''
    def __init__(self, name: str, value: float, **kwargs):
        super().__init__(name, float(value), ndim=0, **kwargs)


@dataclass
class DataRow(BaseData):
    '''
    description:
        A 1D row (vector) of data, ndim fixed at 1.

    args:
        name: identifier for the data (e.g., 'channel_0')
        value: the data itself (1D numpy array of floats)
        metadata: optional dictionary of extra information (units, voxel sizes, axis labels, whatever)
            ***UNDER NO CIRCUMSTANCES SHOULD METADATA BE ACCESSED BY DISPLAY CODE***

    example:
        row = DataRow(name='chan_0', value=np.array([0.1, 0.2, 0.3]))
    '''
    def __init__(self, name: str, value: np.ndarray, **kwargs):
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError("DataRow requires a 1D numpy array")
        super().__init__(name, value, ndim=1, **kwargs)


@dataclass
class DataImage(BaseData):
    '''
    description:
        A 2D image, ndim fixed at 2.

    args:
        name: identifier for the data (e.g., 'channel_0')
        value: the data itself (2D numpy array of floats, ordered (Y, X))
        metadata: optional dictionary of extra information (units, voxel sizes, axis labels, whatever)
            ***UNDER NO CIRCUMSTANCES SHOULD METADATA BE ACCESSED BY DISPLAY CODE***

    example:
        image = DataImage(name='chan_0', value=np.random.rand(512, 512))
    '''
    def __init__(self, name: str, value: np.ndarray, **kwargs):
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            raise ValueError("DataImage requires a 2D numpy array")
        super().__init__(name, value, ndim=2, **kwargs)


@dataclass
class DataVolume(BaseData):
    '''
    description:
        A 3D volume, ndim fixed at 3.

    args:
        name: identifier for the data (e.g., 'channel_0')
        value: the data itself (2D numpy array of floats, ordered (Z, Y, X))
        metadata: optional dictionary of extra information (units, voxel sizes, axis labels, whatever)
            ***UNDER NO CIRCUMSTANCES SHOULD METADATA BE ACCESSED BY DISPLAY CODE***

    example:
        volume = DataVolume(name='chan0', value=np.random.rand(50, 512, 512))
    '''
    def __init__(self, name: str, value: np.ndarray, **kwargs):
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            raise ValueError("DataVolume requires a 3D numpy array")
        super().__init__(name, value, ndim=3, **kwargs)