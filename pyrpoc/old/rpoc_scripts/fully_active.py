import numpy as np

def generate_mask(data_list):
    image = data_list[0]
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[np.asarray(image) > 0] = 255 
    return mask