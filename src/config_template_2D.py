import hashlib
import pickle
import sys
import datetime
from my_discriminators import classical_sgan_disc
from my_generators import classical_sgan_gen
from kgan.losses import *
from kgan.optimizers import Adam


def zx_to_npx(zx, gen_depth):
    '''
    calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx
    '''
    # note: in theano we'd have zx*2**depth
    return zx * (2 ** gen_depth)


class Config:
    def __init__(self, name):

        # Run metadata
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10**8)

        # Training settings
        self.batch_size = 24
        self.epoch_iters = 50
        self.epochs = 50

        # Data dimensions
        self.convdims = 2  # 2D or 3D convolutions
        self.disc_depth = 5
        self.gen_depth = 5
        self.nz = 1  # Number of channels in Z
        self.zx = 12  # Size of each spatial dimensions in Z
        self.zx_sample = 20
        self.nc = 1  # Number of channels
        self.npx = zx_to_npx(self.zx, self.gen_depth)

        # Network setup
        self.discriminator = classical_sgan_disc
        self.generator = classical_sgan_gen
        self.loss_disc_fake = epsilon_gan_disc_fake
        self.loss_disc_true = epsilon_gan_disc_true
        self.loss_gen = epsilon_gan_gen
        self.clip_weights = False  # Clip the discriminator weights (cf Wasserstein GAN)
        self.gradient_penalty = False
        self.c = 0.01  # Clipping value
        self.k = 1  # Number of D updates vs G updates
        self.lmbda = 10  # Gradient penalty factor

        self.disc_args={
            "filter_size": 9,
            "depth": self.disc_depth,
            "channels": self.nc,
            "convdims": self.convdims,
            "clip_weights": self.clip_weights,
            "clipping_value": self.c
        }

        self.gen_args = {
            "filter_size": 5,
            "depth": self.gen_depth,
            "convdims": self.convdims,
            "channels": self.nc
        }

        # Optimizer
        self.optimizer = Adam
        self.lr = 1e-5  # learning rate
        self.b1 = 0.5  # Adam momentum term
        self.optimizer_params = {"lr": self.lr}  # {"lr": self.lr}


if __name__ == "__main__":
    if len(sys.argv) == 2:
        conf = Config(sys.argv[1])
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)

    else:
        conf = Config(str(datetime.datetime.now()))
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)
