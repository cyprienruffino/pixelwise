import hashlib

import models.classical_sgan_disc
import models.classical_sgan_gen

from kgan.losses import *

from runtime.initializers import RandomNormal
from runtime.optimizers import Adam

from config import Config


class CustomConfig(Config):
    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.name = name
        self.seed = None

        # Training settings
        self.batch_size = None
        self.epoch_iters = None
        self.epochs = None
        self.k = None  # Number of D updates vs G updates

        # Optimizer
        self.optimizer = None
        self.optimizer_params = {}

        # Data dimensions
        self.convdims = None  # 2D or 3D convolutions
        self.nz = None  # Number of channels in Z
        self.zx = None  # Size of each spatial dimensions in Z
        self.zx_sample = None
        self.npx = None  # (zx * 2^ depth)

        # Network setup
        self.loss_disc_fake = None
        self.loss_disc_true = None
        self.loss_gen = None

        self.generator = None
        self.gen_args = {}

        self.discriminator = None
        self.disc_args = {}
