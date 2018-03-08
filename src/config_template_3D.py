import hashlib
import pickle
import sys
import datetime
from my_discriminators import classical_sgan_disc
from my_generators import classical_sgan_gen

from kgan.losses import *

from runtime.optimizers import Adam
from runtime import RandomNormal


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
        self.k = 1  # Number of D updates vs G updates

        # Optimizer
        self.optimizer = Adam
        self.lr = 1e-5  # learning rate
        self.b1 = 0.5  # Adam momentum term
        self.optimizer_params = {"lr": self.lr}  # {"lr": self.lr}

        # Data dimensions
        self.convdims = 3  # 2D or 3D convolutions
        self.nz = 1  # Number of channels in Z
        self.zx = 3  # Size of each spatial dimensions in Z
        self.zx_sample = 3
        self.npx = zx_to_npx(self.zx, self.gen_depth)

        # Network setup
        self.discriminator = classical_sgan_disc
        self.generator = classical_sgan_gen
        self.loss_disc_fake = epsilon_gan_disc_fake
        self.loss_disc_true = epsilon_gan_disc_true
        self.loss_gen = epsilon_gan_gen
        self.lmbda = 10  # Gradient penalty factor

        self.disc_args={
            "filter_size": 9,
            "depth": 5,
            "channels": 1,
            "convdims": 3,
            "l2_fac": 1e-5,
            "strides": 2,
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.2),
            "convs": [pow(2, i + 6) for i in range(4)] + [1],
            "clip_weights": False,
            "clipping_value": 0.01
        }

        self.gen_args = {
            "filter_size": 5,
            "depth": 5,
            "channels": 1,
            "convdims": 3,
            "l2_fac": 1e-5,
            "strides": 3,
            "epsilon": 1e-4,
            "init": RandomNormal(stddev=0.2),
            "convs": [pow(2, i + 5) for i in range(4, 0, -1)] + [1]
        }


if __name__ == "__main__":
    if len(sys.argv) == 2:
        conf = Config(sys.argv[1])
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)

    else:
        conf = Config(str(datetime.datetime.now()))
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)
