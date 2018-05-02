import hashlib

import applications.conditional_dilation_only_gen
import applications.conditional_disc
from config import Config
from datasets import data_io2D
from losses import *
from runtime.initializers import RandomNormal
from runtime.optimizers import Adam


class CustomConfig(Config):

    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
                10 ** 8)

        # Training settings
        self.batch_size = 2
        self.epoch_iters = 50
        self.epochs = 150
        self.k = 1  # Number of D updates vs G updates

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

        # Data dimensions
        self.convdims = 2  # 2D or 3D convolutions
        self.nx = 1  # Number of channels in X
        self.nz = 1  # Number of channels in Z
        self.zx = 12  # Size of each spatial dimensions in Z
        self.zx_sample = 12
        self.npx = 384  # (zx * 2^ depth)

        # Network setup
        self.loss_disc_fake = gan_disc_fake
        self.loss_disc_true = gan_disc_true
        self.loss_gen = gan_gen

        self.generator = applications.upscaling_dilation_gen.create_network
        self.gen_args = {
            "filter_size": 5,
            "convdims": 2,
            "channels": self.nz,
            "l2_fac": 1e-5,
            "upscaling_filters": [512, 256, 128, 64, 1],
            "strides": [2, 2, 2, 2, 2],
            "dilations_filters": [64, 128, 256, 512, 1],
            "dilations": [1, 2, 3, 4, 5],
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.02)
        }

        self.discriminator = applications.classical_sgan_disc.create_network
        self.disc_args = {
            "filter_size": 9,
            "filters": [64, 128, 256, 512, 1],
            "channels": self.nx,
            "convdims": 2,
            "l2_fac": 1e-5,
            "strides": [2, 2, 2, 2, 2],
            "alpha": 0.2,
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.02)
        }

        self.disc_data_provider = data_io2D.disc_data_provider
        self.disc_data_provider_args = {
            "zx": self.zx,
            'npx': self.npx,
            "batch_size": self.batch_size,
            "filter": None,
            "mirror": True,
            "nx": self.nx,
        }

        self.gen_data_provider = data_io2D.gen_data_provider
        self.gen_data_provider_args = {
            "zx": self.zx,
            "batch_size": self.batch_size,
            "nz": self.nz,
        }
