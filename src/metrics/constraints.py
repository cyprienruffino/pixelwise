import numpy as np


def mse(fake, const):
    mask = np.greater(np.abs(const), 0).astype(float)
    return np.sum(np.square(const - fake*mask)) / (np.sum(mask) + 1e-8)