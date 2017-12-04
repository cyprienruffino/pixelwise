import hashlib
import pickle
import sys
import datetime

from .factory import discriminators, generators, losses, optimizers


class Config:
    def __init__(self, name):

        # Run metadata
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10**8)

        # Training settings
        self.batch_size = 24
        self.epoch_iters = self.batch_size * 100
        self.epochs = 40
        self.k = 5  # Number of D updates vs G updates

        # Data dimensions
        self.convdims = 2  # 2D or 3D convolutions
        self.nz = 1  # Number of channels in Z
        self.zx = 12  # Size of each spatial dimensions in Z
        self.zx_sample = 12
        self.nc = 1  # Number of channels

        # Network setup
        self.discriminator = discriminators.classical_sgan
        self.generator = generators.classical_sgan
        self.losses = losses.wasserstein_gan
        self.clip_weights = True  # Clip the discriminator weights (cf Wasserstein GAN)
        self.c = 0.01  # Clipping value

        # Optimizer
        self.optimizer = optimizers.adam
        self.lr = 0.0005  # learning rate
        self.b1 = 0.5  # Adam momentum term
        self.optimizer_params = {"lr": self.lr, "beta_1": self.b1}


if __name__ == "__main__":
    if len(sys.argv) == 2:
        conf = Config(sys.argv[1])
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)

    else:
        conf = Config(str(datetime.datetime.now()))
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)
