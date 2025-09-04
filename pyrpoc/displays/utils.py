import numpy as np

def make_gray_to_red_colormap():
    # 0.0 to just below 1.0: grayscale
    # 1.0: red
    positions = np.array([0.0, 0.99, 1.0])
    colors = np.array([
        [0, 0, 0, 255],      # black
        [255, 255, 255, 255],# white
        [255, 0, 0, 255],    # red
    ])
    return pg.ColorMap(positions, colors)