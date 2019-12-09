import hashlib

import tensorflow as tf

import metrics.constraints
from config import Config

from applications import dcgan_cifar10
from datasets import cifar10
from utils import noise, synthetic

class CustomConfig(Config):

    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10 ** 8)

        # Training settings
        self.batch_size = 32
        self.epochs = 50
        self.k = 1  # Number of D updates vs G updates
        self.lmbda = 100
        self.validation = True
        self.test = True

        # Optimizers
        self.disc_optimizer = tf.train.AdamOptimizer
        self.disc_optimizer_args = {
            "learning_rate": 0.0002,
            "beta1": 0.5
        }
        self.gen_optimizer = tf.train.AdamOptimizer
        self.gen_optimizer_args = {
            "learning_rate": 0.0002,
            "beta1": 0.5
        }

        # Data dimensions
        self.channels = 3
        self.nz = 1  # Number of channels in Z
        self.zx = 4  # Size of each spatial dimensions in Z
        self.zx_sample = 4
        self.npx = 32  # (zx * 2^ depth)

        self.dataset_size = 42500
        self.valid_size = 2500
        self.test_size = 9000

        # Network setup
        self.generator = dcgan_cifar10.create_gen
        self.gen_args = {
            "img_shape": (self.npx, self.npx, self.channels),
            "noise_shape": (self.zx, self.zx, self.channels)
        }

        self.discriminator = dcgan_cifar10.create_disc
        self.disc_args = {
            "img_shape": (self.npx, self.npx, self.channels)
        }

        self.noise_provider = noise.uniform
        self.noise_provider_args = {
            "zx": self.zx,
            "nz": self.nz
        }

        self.fid_model = "fid_models/cifar10.hdf5"
