import numpy as np


def uniform(zx, convdims=2, nz=1, start=-1., stop=1.):
    def _noise(batch_size):
        return np.random.uniform(start, stop, (batch_size,) + (zx,) * convdims + (nz,))
    return _noise


def normal(zx, convdims=2, nz=1, mu=0, sigma=1):
    def _noise(batch_size):
        return np.random.normal(mu, sigma, (batch_size,) + (zx,) * convdims + (nz,))
    return _noise
