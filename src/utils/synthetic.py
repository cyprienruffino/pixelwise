import numpy as np


def bernoulli(p=0.01):
    def get_const(batch):
        return batch * np.random.binomial(1, p, size=np.prod(batch.shape)).reshape(batch.shape)
    return get_const


def exact(p=0.01):
    def get_const(batch):
        out = np.zeros(batch.shape)

        for img in range(len(batch)):
            n = int(np.size(batch[img]) * p)
            x = np.random.choice(np.arange(batch[img].shape[1]), n)
            y = np.random.choice(np.arange(batch[img].shape[2]), n)
            out[img, x, y] = batch[img, x, y]

        return out

    return get_const
