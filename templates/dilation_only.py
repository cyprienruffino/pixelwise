import hashlib
import models.dilation_only_gen
import models.conditional_disc

from kgan.losses import *

from runtime.initializers import RandomNormal
from runtime.optimizers import Adam

from config import Config


class CustomConfig(Config):
    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
                10 ** 8)

        # Training settings
        self.batch_size = 24
        self.epoch_iters = 50
        self.epochs = 500
        self.k = 1  # Number of D updates vs G updates

        # Optimizer
        self.optimizer = Adam
        self.optimizer_params = {
            "lr": 0.0005,
            "beta_1": 0.5
        }

        # Data dimensions
        self.convdims = 2  # 2D or 3D convolutions
        self.nz = 1  # Number of channels in Z
        self.zx = 12  # Size of each spatial dimensions in Z
        self.zx_sample = 20
        self.npx = 384  # (zx * 2^ depth)

        # Network setup
        self.loss_disc_fake = gan_disc_fake
        self.loss_disc_true = gan_disc_true
        self.loss_gen = gan_gen

        self.generator = models.dilation_only_gen.create_network
        self.gen_args = {
            "filter_size": 5,
            "convdims": 2,
            "channels": 1,
            "l2_fac": 1e-5,
            "filters": [64, 64, 128, 128, 256, 256, 512, 1],
            "dilations": [1, 2, 2, 3, 3, 4, 1],
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.02)
        }

        self.discriminator = models.conditional_disc.create_network
        self.disc_args = {
            "filter_size": 9,
            "filters": [64, 128, 256, 512, 1],
            "channels": 1,
            "convdims": 2,
            "l2_fac": 1e-5,
            "strides": [2, 2, 2, 2, 2],
            "alpha": 0.2,
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.02)
        }
