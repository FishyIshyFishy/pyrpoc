from __future__ import annotations
import numpy as np

from pyrpoc.utils import DataImage, BaseParameter, AcquisitionContext, ModalityContext, DataContext
from pyrpoc.utils.base_types import BaseLaserModulation, BaseModality, BaseInstrument, modality_registry


@modality_registry.register('simulated')
class SimulatedModality(BaseModality):
    '''
    description:
        A mock modality used for testing the acquisition pipeline.
        Generates synthetic image data without requiring instruments.
    '''

    required_parameters: list[BaseParameter] = [
    ]

    required_instruments: list[type[BaseInstrument]] = []
    allowed_modulations: list[type[BaseLaserModulation]] = []
    emission_data_type = DataImage

    def __init__(self, context: AcquisitionContext):
        super().__init__(context)


    def perform_acquisition(self) -> DataImage:
        '''
        description:
            Generates a single synthetic image frame using random noise.
        '''
        x = int(self.acquisition_params.get('x_pixels', 256))
        y = int(self.acquisition_params.get('y_pixels', 256))
        avg = float(self.acquisition_params.get('average_value', 0.5))
        std = float(self.acquisition_params.get('std_value', 0.1))

        img = np.random.normal(loc=avg, scale=std, size=(y, x))
        img = np.clip(img, 0, 1)

        return DataImage(name='ch1', value=img)


