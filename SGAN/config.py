import hashlib
import pickle
import sys
import datetime

from enum import Enum


def zx_to_npx(zx, depth):
    '''
    calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx
    '''
    # note: in theano we'd have zx*2**depth
    return (zx - 1) * 2**depth + 1


class Losses(Enum):
    classical_gan = "classical_gan"
    epsilon_gan = "epsilon_gan"
    softplus_gan = "softplus_gan"
    wasserstein_gan = "wasserstein_gan"


class GenUpscaling(Enum):
    deconvolution = "deconvolution"
    upsampling = "upsampling"


class Config:
    def __init__(self, name):

        self.name = name

        # Network setup
        # GAN or Wasserstein GAN

        self.losses = Losses.softplus_gan
        self.clip_gradients = False
        self.noise = True
        self.c = 0.01

        self.gen_up = GenUpscaling.deconvolution

        # Depth
        self.gen_depth = 5
        self.dis_depth = 5

        # Data dimensions
        self.convdims = 2  # 2D or 3D convolutions
        self.nz = 1  # num of dim for Z at each field position (d in the paper)
        self.zx = 12  # num of spatial dimensions in Z (l and m in the paper)
        self.zx_sample = 20  # size of the spatial dimension in Z
        # num of pixels width/height of images in X
        self.npx = zx_to_npx(self.zx, self.gen_depth)

        # Kernels
        self.gen_ks = [5] * self.gen_depth
        self.dis_ks = [9] * self.dis_depth
        self.nc = 1  # Number of channels

        # Number of filters
        self.gen_fn = (
            [self.nc] + [2**(n + 6) for n in range(self.gen_depth - 1)])[::-1]
        self.dis_fn = [2**(n + 6) for n in range(self.dis_depth - 1)] + [1]

        # Strides
        self.gen_strides = [2] * self.gen_depth
        self.dis_strides = [2] * self.dis_depth

        # Hyperparameters
        self.lr = 0.0005  # learning rate of adam
        self.b1 = 0.5  # momentum term of adam
        self.l2_fac = 1e-5  # L2 weight regularization factor

        # Seed, for reproductibility. Set np.random.seed with this
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10**8)

        # Learning parameters
        self.batch_size = 64
        self.epoch_iters = self.batch_size * 100
        self.epochs = 50
        self.k = 1  # Number of D updates vs G updates


if __name__ == "__main__":
    if len(sys.argv) == 2:
        conf = Config(sys.argv[1])
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)

    else:
        conf = Config(str(datetime.datetime.now()))
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)
