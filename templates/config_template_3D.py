import hashlib

import models.classical_sgan_disc
import models.classical_sgan_gen

from config import Config

from kgan.losses import *

from runtime.initializers import RandomNormal
from runtime.optimizers import Adam


class CustomConfig(Config):
    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10**8)

        # Training settings
        self.batch_size = 24
        self.epoch_iters = 50
        self.epochs = 50
        self.k = 1  # Number of D updates vs G updates

        # Optimizer
        self.optimizer = Adam
        self.optimizer_params = {
            "lr": 2e-4,
            "beta_1": 0.5
        }

        # Data dimensions
        self.convdims = 3  # 2D or 3D convolutions
        self.nz = 1  # Number of channels in Z
        self.zx = 3  # Size of each spatial dimensions in Z
        self.zx_sample = 3
        self.npx = 96  # zx * 2^depth

        # Network setup
        self.loss_disc_fake = gan_disc_fake
        self.loss_disc_true = gan_disc_true
        self.loss_gen = gan_gen

        self.generator = models.classical_sgan_gen.create_network
        self.gen_args = {
            "filter_size": 5,
            "filters": [pow(2, i + 5) for i in range(4, 0, -1)] + [1],
            "depth": 5,
            "channels": 1,
            "convdims": 3,
            "l2_fac": 1e-5,
            "strides": [2, 2, 2, 2, 2],
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.2),
        }

        self.discriminator = models.classical_sgan_disc.create_network
        self.disc_args={
            "filter_size": 9,
            "filters": [pow(2, i + 6) for i in range(4)] + [1],
            "depth": 5,
            "channels": 1,
            "convdims": 3,
            "l2_fac": 1e-5,
            "strides": [2, 2, 2, 2, 2],
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.2),
            "clip_weights": False,
            "clipping_value": 0.01
        }
