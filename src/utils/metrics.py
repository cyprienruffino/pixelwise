import numpy as np
import skimage.feature


def lbp(real, fake, npoints=1, radius=1):
    real_lbp = []
    fake_lbp = []

    for data in real:
        real_lbp.append(skimage.feature.local_binary_pattern(np.squeeze(data), npoints, radius))

    for data in fake:
        fake_lbp.append(skimage.feature.local_binary_pattern(np.squeeze(data), npoints, radius))

    return np.sum(np.square(np.subtract(fake_lbp, real_lbp)))


def hog(real, fake):
    real_lbp = []
    fake_lbp = []

    for data in real:
        real_lbp.append(skimage.feature.hog(np.squeeze(data)))

    for data in fake:
        fake_lbp.append(skimage.feature.hog(np.squeeze(data)))

    return np.sum(np.square(np.subtract(fake_lbp, real_lbp)))


def _tvaniso(img):
    dx = np.abs(img[:, 1:] - img[:, :-1])
    dy = np.abs(img[1:, :] - img[:-1, :])
    return dx.mean() + dy.mean()


def tvaniso(real, fake):
    tv_real = np.array([_tvaniso(np.squeeze(i)) for i in real])
    tv_fake = np.array([_tvaniso(np.squeeze(i)) for i in fake])
    return np.sum(np.square(tv_real - tv_fake))


def _tviso(img):
    dx = np.abs(img[:-1, 1:] - img[:-1, :-1])
    dy = np.abs(img[1:, :-1] - img[:-1, :-1])
    return np.sqrt(dx ** 2 + dy ** .2).mean()


def tviso(real, fake):
    tv_real = np.array([_tviso(np.squeeze(i)) for i in real])
    tv_fake = np.array([_tviso(np.squeeze(i)) for i in fake])
    return np.sum(np.square(tv_real - tv_fake))


def mse(real, fake, mask):
    return np.sum(np.square(real*mask - fake*mask)) / np.sum(mask)
