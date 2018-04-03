import hashlib

import applications.classical_sgan_disc
import applications.classical_sgan_gen

from config import Config
from datasets import data_io3D

from losses import *

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
        self.epoch_iters = 100
        self.epochs = 50
        self.k = 1  # Number of D updates vs G updates

        # Optimizer
        self.optimizer = Adam
        self.optimizer_params = {
            "lr": 5e-4,
            "beta_1": 0.5
        }

        # Data dimensions
        self.convdims = 3  # 2D or 3D convolutions
        self.nz = 1  # Number of channels in Z
        self.zx = 3  # Size of each spatial dimensions in Z
        self.zx_sample = 4
        self.npx = 96  # zx * 2^depth

        # Network setup
        self.loss_disc_fake = gan_disc_fake
        self.loss_disc_true = gan_disc_true
        self.loss_gen = gan_gen

        self.generator = models.classical_sgan_gen.create_network
        self.gen_args = {
            "filter_size": 5,
            "filters": [pow(2, i + 5) for i in range(4, 0, -1)] + [1],
            "channels": 1,
            "convdims": 3,
            "l2_fac": 1e-5,
            "strides": [2, 2, 2, 2, 2],
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.2),
        }

        self.discriminator = models.classical_sgan_disc.create_network
        self.disc_args={
            "filter_size": 5,
            "filters": [pow(2, i + 6) for i in range(4)] + [1],
            "channels": 1,
            "convdims": 3,
            "l2_fac": 1e-5,
            "strides": [2, 2, 2, 2, 2],
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.02),
            "clip_weights": False,
            "clipping_value": 0.01
        }

        self.data_generator = data_io3D.get_texture_iter
        self.data_gen_args = {
            'npx': self.npx,
            "batch_size": self.batch_size,
            "filter": None,
            "mirror": True,
            "n_channel": self.nz,
        }