import hashlib

import tensorflow as tf

from applications import upscaling_dilation_gen
from applications import disc
from config import Config
from datasets import data_io2D


class CustomConfig(Config):

    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
                10 ** 8)

        # Training settings
        self.batch_size = 2
        self.epoch_iters = 100
        self.epochs = 100
        self.k = 1  # Number of D updates vs G updates

        # Optimizers
        self.disc_optimizer = tf.train.AdamOptimizer
        self.disc_optimizer_args = {
            "learning_rate": 0.00005,
            "beta1": 0.5
        }
        self.gen_optimizer = tf.train.AdamOptimizer
        self.gen_optimizer_args = {
            "learning_rate": 0.00005,
            "beta1": 0.5
        }

        # Data dimensions
        self.channels = 1
        self.nz = 1  # Number of channels in Z
        self.zx = 12  # Size of each spatial dimensions in Z
        self.zx_sample = 12
        self.npx = 384  # (zx * 2^ depth)

        self.constraints_ratio = 0.001
        self.lmbda = 50

        # Network setup
        self.generator = upscaling_dilation_gen.create_network
        self.gen_args = {
            "filter_size": 5,
            "channels": self.channels,
        }

        self.discriminator = disc.create_network
        self.disc_args = {
            "filter_size": 9,
            "channels": self.channels,
        }
        self.data_provider = data_io2D.get_data_generator
        self.data_provider_args = {
            "batch_size": self.batch_size,
            "npx": self.npx,
            "n_channel": self.channels
        }
