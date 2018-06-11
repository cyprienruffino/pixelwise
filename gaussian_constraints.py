import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

sigma, mu = 2, 0
const = 100

nb = 64

grid = np.zeros((nb, 200, 200))

for i in range(nb):
    signs = np.asarray([-1 if x==0 else 1 for x in np.random.randint(0, 2, const)])
    xs = np.random.randint(0, len(grid[1]), const)
    ys = np.random.randint(0, len(grid[1]), const)
    grid[i, xs, ys] = signs

output = np.zeros(grid.shape)

for i in range(len(posgrid)):
    blur = gaussian_filter(grid[i], sigma=sigma, cval=1)
    output[i] = blur / np.min(blur)

plt.imshow(output[0])

plt.imshow(grid[0])
grid.min()
grid.max()
