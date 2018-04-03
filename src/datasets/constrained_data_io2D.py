import numpy as np
from scipy.stats import bernoulli

from datasets import data_io2D


def get_noise(batch_size, zx, nz, convdims):
        return np.random.uniform(-1., 1., (batch_size, nz) + ((zx,) * convdims))


def get_constraints(batch, constraints_ratio=0.1):
    masks = bernoulli.rvs(constraints_ratio, size=batch.size).reshape(batch.shape)
    return batch * masks


def gen_data_provider(folder, batch_size, npx, zx, convdims=2, constraints_ratio=0.1, filter=False, mirror=True, n_channel=1):
    generator = data_io2D.get_texture_iter(folder + "/", npx, batch_size, filter, mirror, n_channel)
    while True:
        data = next(generator)
        noise = get_noise(batch_size, zx, n_channel, convdims)
        constraints = get_constraints(data, constraints_ratio)
        yield [noise, constraints]


def disc_data_provider(folder, batch_size, npx, zx, convdims=2, constraints_ratio=0.1, filter=False, mirror=True, n_channel=1):
    generator = data_io2D.get_texture_iter(folder + "/", npx, batch_size, filter, mirror, n_channel)
    while True:
        data = next(generator)
        noise = get_noise(batch_size, zx, n_channel, convdims)
        constraints = get_constraints(data, constraints_ratio)
        yield [data, constraints]