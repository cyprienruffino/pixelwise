import hashlib

import tensorflow as tf

import metrics.fid
from config import Config

from applications import unet as gen
from applications import disc_patchgan_nocond as disc
from datasets import from_hdf5
from utils import noise, synthetic


class CustomConfig(Config):

    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
                10 ** 8)

        # Training settings
        self.batch_size = 16
        self.epochs = 50
        self.k = 1  # Number of D updates vs G updates
        self.lmbda = 0.1
        self.validation = True
        self.test = True

        # Optimizers
        self.disc_optimizer = tf.train.AdamOptimizer
        self.disc_optimizer_args = {
            "learning_rate": 0.00001,
            "beta1": 0.5
        }
        self.gen_optimizer = tf.train.AdamOptimizer
        self.gen_optimizer_args = {
            "learning_rate": 0.00001,
            "beta1": 0.5
        }

        # Data dimensions
        self.channels = 3
        self.nz = 3  # Number of channels in Z
        self.zx = 40  # Size of each spatial dimensions in Z
        self.npx = 160  # (zx * 2^ depth)
        self.dataset_size = 20000
        self.valid_size = 2000
        self.test_size = 4000

        # Network setup
        self.generator = gen.create_network
        self.gen_args = {
            "channels": 3
        }

        self.discriminator = disc.create_network
        self.disc_args = {
            "img_shape": (self.npx, self.npx, self.channels),
        }

        # Noise
        self.noise_provider = noise.uniform
        self.noise_provider_args = {
            "zx": self.zx,
            "nz": self.nz
        }

        self.fid_model = "fid_models/dtd_160.hdf5"
        self.metrics = {
        }
