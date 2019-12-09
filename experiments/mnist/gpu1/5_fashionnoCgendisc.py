import hashlib

import tensorflow as tf

import metrics.fid
from config import Config

from applications import dcgan_mnist_noc
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
        self.batch_size = 32
        self.epochs = 50
        self.k = 1  # Number of D updates vs G updates
        self.lmbda = 0
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
        self.channels = 1
        self.nz = 1  # Number of channels in Z
        self.zx = 7  # Size of each spatial dimensions in Z
        self.npx = 28  # (zx * 2^ depth)
        self.dataset_size = 52000
        self.valid_size = 8000
        self.test_size = 10000

        # Network setup
        self.generator = dcgan_mnist_noc.create_gen
        self.gen_args = {
            "img_shape": (self.npx, self.npx, self.channels),
            "noise_shape": (self.zx, self.zx, self.channels)
        }

        self.discriminator = dcgan_mnist_noc.create_disc
        self.disc_args = {
            "img_shape": (self.npx, self.npx, self.channels)
        }

        # Noise
        self.noise_provider = noise.uniform
        self.noise_provider_args = {
            "zx": self.zx,
            "nz": self.nz
        }

        self.fid_model = "fid_models/fashion.hdf5"
        self.metrics = {
            "TVI": metrics.tvi,
            "TVA": metrics.tva,
            "LBP": metrics.lbp,
            "HOG": metrics.hog
        }
