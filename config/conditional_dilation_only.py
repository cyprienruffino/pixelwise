import hashlib

import applications.conditional_disc
import applications.conditional_dilation_only_gen
from config import Config
from datasets import constrained_data_io2D
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
        self.epochs = 50
        self.k = 1  # Number of D updates vs G updates

        # Optimizers
        self.disc_optimizer = Adam
        self.disc_optimizer_args = {
            "lr": 0.0001,
            "beta_1": 0.5
        }
        self.gen_optimizer = Adam
        self.gen_optimizer_args = {
            "lr": 0.0001,
            "beta_1": 0.5
        }

        # Data dimensions
        self.convdims = 2  # 2D or 3D convolutions
        self.nz = 1  # Number of channels in Z
        self.zx = 384  # Size of each spatial dimensions in Z
        self.zx_sample = 384
        self.npx = 384  # (zx * 2^ depth)

        # Network setup
        self.loss_disc_fake = gan_disc_fake
        self.loss_disc_true = gan_disc_true
        self.loss_gen = gan_gen

        self.generator = applications.conditional_dilation_only_gen.create_network
        self.gen_args = {
            "filter_size": 5,
            "convdims": 2,
            "channels": 1,
            "l2_fac": 1e-5,
            "filters": [64, 128, 128, 256, 256, 512, 512, 1],
            "dilations": [1, 1, 2, 2, 3, 3, 1],
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.02)
        }

        self.discriminator = applications.conditional_disc.create_network
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

        self.disc_data_provider = constrained_data_io2D.disc_data_provider
        self.disc_data_provider_args = {
            "zx": self.zx,
            'npx': self.npx,
            "batch_size": self.batch_size,
            "constraints_ratio": 0.1,
            "filter": None,
            "mirror": True,
            "n_channel": self.nz,
        }

        self.gen_data_provider = constrained_data_io2D.gen_data_provider
        self.gen_data_provider_args = {
            "zx": self.zx,
            'npx': self.npx,
            "batch_size": self.batch_size,
            "filter": None,
            "mirror": True,
            "n_channel": self.nz,
            "constraints_ratio": 0.1
        }
