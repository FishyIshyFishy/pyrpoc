import numpy as np
import matplotlib.pyplot as plt
import tifffile

file_path = r'C:\Users\ishaa\Documents\ZhangLab\python\new_pysrs\test_data\acquisition_raw.npz'

with np.load(file_path, allow_pickle=True) as data:
    print("Keys in the file:", data.files)
    
    # Accessing inside the block works!
    params = data['acquisition_parameters']
    frames = data['frames']
    
    print(params)
    print(f"Frames shape: {frames.shape}")

frames = tifffile.imread(r'C:\Users\ishaa\Documents\ZhangLab\python\new_pysrs\test_data\acquisition_intensity.tiff')
plt.imshow(frames)
plt.show()