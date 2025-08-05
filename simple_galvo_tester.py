import numpy as np
import matplotlib.pyplot as plt
from pyrpoc.instruments.galvo import Galvo

def test_parameter_variations():
    """Test different parameter combinations"""
    galvo = Galvo(name="Test Galvo")
    galvo.parameters = {
        'slow_axis_channel': 0,
        'fast_axis_channel': 1,
        'sample_rate': 10000000,
        'device_name': 'Dev1'
    }
    
    # Test different configurations
    test_configs = [
        {
            'name': '2k',
            'params': {
                'dwell_time': 1,
                'extrasteps_left': 100,
                'extrasteps_right': 20,
                'amplitude_x': 3.0,
                'amplitude_y': 3.0,
                'offset_x': -1.5,
                'offset_y': 0.0,
                'x_pixels': 300,
                'y_pixels': 100
            }
        },
        
    ]
    
    for i, config in enumerate(test_configs):
        waveform = galvo.generate_raster_waveform(config['params'])
        plt.plot(waveform[0])
        plt.plot(waveform[1])
        plt.show()

if __name__ == "__main__":
   test_parameter_variations()