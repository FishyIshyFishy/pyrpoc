import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

def gen_comet(size, comets, movement=None, noise=0.1):
    frame = np.zeros(size)
    for i, (x,y) in enumerate(comets):
        if movement:
            x += movement[i][0]
            y += movement[i][1]
        x = int(np.clip(x, 0, size[1]-1))
        y = int(np.clip(y, 0, size[0]-1))
        frame[y, x] += 1
    frame = gaussian_filter(frame, sigma=2)
    frame += np.random.normal(0, noise, size)
    return np.clip(frame, 0, 1)

size = (256,256)
comets = 50
max_mv = 20
noise = 0

pos = [(np.random.uniform(0, size[1]), np.random.uniform(0, size[0])) for _ in range(comets)]
vec_alive = [(np.random.uniform(-max_mv, max_mv), np.random.uniform(-max_mv, max_mv)) for _ in range(comets)]
vec_dead = [(0, 0) for _ in range(comets)]

alive1 = gen_comet(size, pos, noise=noise)
alive2 = gen_comet(size, pos, movement=vec_alive, noise=noise)
dead1 = gen_comet(size, pos, noise=noise)
dead2 = gen_comet(size, pos, movement=vec_dead, noise=noise)

flow_alive = cv2.calcOpticalFlowFarneback((alive1*255).astype(np.uint8), (alive2*255).astype(np.uint8), None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow_dead = cv2.calcOpticalFlowFarneback((dead1*255).astype(np.uint8), (dead2*255).astype(np.uint8), None, 0.5, 3, 15, 3, 5, 1.2, 0)

mag_alive = np.linalg.norm(flow_alive, axis=2)
mag_dead = np.linalg.norm(flow_dead, axis=2)

fig, ax = plt.subplots(2,3, figsize=(15,10))
ax[0,0].imshow(alive1, cmap='gray')
ax[0,0].set_title('first alive frame')
ax[0,1].imshow(alive2, cmap='gray')
ax[0,1].set_title('2nd alive frame')
ax[0,2].imshow(mag_alive, cmap='magma')
ax[0,2].set_title('optical flow vector magnitude for alive')

ax[1,0].imshow(dead1, cmap='gray')
ax[1,0].set_title('dead 1')
ax[1,1].imshow(dead2, cmap='gray')
ax[1,1].set_title('dead 2')
ax[1,2].imshow(mag_dead, cmap='magma')
ax[1,2].set_title('same for dead')

for a in ax.flat:
    a.axis('off')

plt.tight_layout()
plt.show()