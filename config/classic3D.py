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

        # Data dimensions
        self.convdims = 3  # 2D or 3D convolutions
        self.nx = 1  # Number of channels in X
        self.nz = 1  # Number of channels in Z
        self.zx = 3  # Size of each spatial dimensions in Z
        self.zx_sample = 4
        self.npx = 96  # zx * 2^depth

        # Networks
        self.generator = applications.classical_sgan_gen.create_network
        self.gen_args = {
            "filter_size": 5,
            "filters": [pow(2, i + 5) for i in range(4, 0, -1)] + [1],
            "channels": self.nz,
            "convdims": 3,
            "l2_fac": 1e-5,
            "strides": [2, 2, 2, 2, 2],
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.2),
        }

        self.discriminator = applications.classical_sgan_disc.create_network
        self.disc_args={
            "filter_size": 5,
            "filters": [pow(2, i + 6) for i in range(4)] + [1],
            "channels": self.nx,
            "convdims": 3,
            "l2_fac": 1e-5,
            "strides": [2, 2, 2, 2, 2],
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.02),
            "clip_weights": False,
            "clipping_value": 0.01
        }

        # Optimizers
        self.disc_optimizer = Adam
        self.disc_optimizer_args = {
            "lr": 0.0005,
            "beta_1": 0.5
        }
        self.gen_optimizer = Adam
        self.gen_optimizer_args = {
            "lr": 0.0005,
            "beta_1": 0.5
        }

        # Losses
        self.loss_disc_fake = gan_disc_fake
        self.loss_disc_true = gan_disc_true
        self.loss_gen = gan_gen

        # Data providers
        self.disc_data_provider = data_io3D.disc_data_provider
        self.disc_data_provider_args = {
            "zx": self.zx,
            'npx': self.npx,
            "convdims": self.convdims,
            "batch_size": self.batch_size,
            "filter": None,
            "mirror": True,
            "nx": self.nx,
        }

        self.gen_data_provider = data_io3D.gen_data_provider
        self.gen_data_provider_args = {
            "zx": self.zx,
            "batch_size": self.batch_size,
            "convdims": self.convdims,
            "nz": self.nz
        }