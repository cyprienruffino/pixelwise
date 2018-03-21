import numpy as np
from scipy.stats import bernoulli

from datasets.data_io2D import get_texture_iter as generator2D


def get_texture_iter(folder, npx=128, batch_size=64, \
                     filter=None, mirror=True, n_channel=1, constraints_ratio=0.1):

    generator = generator2D(folder, npx, batch_size, filter, mirror, n_channel)

    while True:
        batch = next(generator)
        masks = bernoulli.rvs(constraints_ratio, size=np.prod(batch.shape)[0]).reshape(batch.shape)

        yield [batch, batch*masks]
