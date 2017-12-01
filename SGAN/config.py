import hashlib
import pickle
import sys
import datetime

from enum import Enum



class Losses(Enum):
    classical_gan = "classical_gan"
    epsilon_gan = "epsilon_gan"
    softplus_gan = "softplus_gan"
    wasserstein_gan = "wasserstein_gan"
    wasserstein_min_gan = "wasserstein_min_gan"


class Optimizer(Enum):
    adam = "adam"
    rmsprop = "rmsprop"
    sgd = "sgd"


class Config:
    def __init__(self, name):
        self.name = name

        # Hyperparameters
        self.optimizer = Optimizer.adam
        self.lr = 0.0005  # learning rate
        self.b1 = 0.5  # Adam momentum term
        self.momentum = 0.9  # SGD momentum term
        self.batch_size = 24
        self.epoch_iters = self.batch_size * 100
        self.epochs = 40
        self.k = 1  # Number of D updates vs G updates

        # Seed, for reproductibility. Set np.random.seed with this
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10**8)

        # Network setup
        # GAN or Wasserstein GAN

        self.losses = Losses.wasserstein_gan
        self.clip_weights = False
        self.clip_gradients = False
        self.c = 0.01

        # Data dimensions
        self.convdims = 2  # 2D or 3D convolutions
        self.nz = 1  # num of dim for Z at each field position (d in the paper)
        self.zx = 12  # num of spatial dimensions in Z (l and m in the paper)
        self.zx_sample = 12  # size of the spatial dimension in Z
        # num of pixels width/height of images in X
        self.nc = 1  # Number of channels


if __name__ == "__main__":
    if len(sys.argv) == 2:
        conf = Config(sys.argv[1])
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)

    else:
        conf = Config(str(datetime.datetime.now()))
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)
