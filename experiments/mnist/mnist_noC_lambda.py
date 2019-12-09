import hashlib

import tensorflow as tf

import metrics.fid
from config import Config

from applications import dcgan_mnist_noc
from datasets import mnist
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
        self.epoch_iters = 2000
        self.epochs = 40
        self.k = 1  # Number of D updates vs G updates

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

        self.lmbda = 0

        # Network setup
        self.generator = dcgan_mnist_noc.create_gen
        self.gen_args = {
            "img_shape": (self.npx, self.npx, self.channels),
            "noise_shape": (self.zx, self.zx, self.channels)
        }

        self.discriminator = dcgan_mnist_noc.create_disc_noC
        self.disc_args = {
            "img_shape": (self.npx, self.npx, self.channels)
        }

        # Dataset
        self.dataset_size = mnist.TRAIN_SIZE

        self.data_provider = mnist.get_data_provider
        self.data_provider_args = {"batch_size": self.batch_size}

        self.gen_const_provider = mnist.get_genconst_provider
        self.gen_const_provider_args = {"batch_size": self.batch_size}

        # Valid set
        self.valid_size = mnist.VALID_SIZE

        self.valid_data_provider = mnist.get_valid_provider
        self.valid_constraints_provider = mnist.get_validconst_provider

        self.valid_data_provider_args = {"batch_size": self.batch_size}
        self.valid_constraints_provider_args = {"batch_size": self.batch_size}

        # Test set
        self.test_size = mnist.TEST_SIZE
        self.test_data_provider = mnist.get_test_provider
        self.test_constraints_provider = mnist.get_testconst_provider

        self.test_data_provider_args = {"batch_size": self.batch_size}
        self.test_constraints_provider_args = {"batch_size": self.batch_size}

        # Noise
        self.noise_provider = noise.uniform
        self.noise_provider_args = {
            "zx": self.zx,
            "nz": self.nz
        }

        self.metrics = {
            "TVI": metrics.tvi,
            "TVA": metrics.tva,
            "LBP": metrics.lbp,
            "HOG": metrics.hog
        }
