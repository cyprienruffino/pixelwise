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
        self.batch_size = 25
        self.epoch_iters = (64 - self.batch_size) * 100
        self.epochs = 50

        # Data dimensions
        self.convdims = 3  # 2D or 3D convolutions
        self.disc_depth = 3
        self.gen_depth = 3
        self.nz = 1  # Number of channels in Z
        self.zx = 4  # Size of each spatial dimensions in Z
        self.zx_sample = 5
        self.nc = 1  # Number of channels
        self.npx = zx_to_npx(self.zx, self.gen_depth)

        # Network setup
        self.discriminator = classical_sgan_disc
        self.generator = classical_sgan_gen
        self.loss_disc_fake = wasserstein_disc_fake
        self.loss_disc_true = wasserstein_disc_true
        self.loss_gen = wasserstein_gen
        self.clip_weights = False  # Clip the discriminator weights (cf Wasserstein GAN)
        self.gradient_penalty = True
        self.c = 0.01  # Clipping value
        self.k = 5  # Number of D updates vs G updates
        self.lmbda = 10  # Gradient penalty factor

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
