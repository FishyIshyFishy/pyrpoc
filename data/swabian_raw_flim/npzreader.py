import numpy as np

data = np.load(r'C:\Users\ishaa\Documents\ZhangLab\python\new_pysrs\data\swabian_raw_flim\20260304_123247_flim_test_001_raw.npz',
               allow_pickle=True)
print(data.files)
frames = data['frames']
print(frames.shape)