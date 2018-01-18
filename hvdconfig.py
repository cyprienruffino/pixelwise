import hashlib
import pickle
import sys
import datetime
from my_discriminators import classical_sgan_disc
from my_generators import classical_sgan_gen
from kgan.losses import epsilon_gan_fake, epsilon_gan_true, epsilon_gan_gen
from kgan.optimizers import Adam


def zx_to_npx(zx, depth):
    '''
    calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx
    '''
    # note: in theano we'd have zx*2**depth
    return zx * (2 ** depth)


class Config:
    def __init__(self, name):

        # Run metadata
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10**8)

        # Training settings
        self.batch_size = 32
        self.epoch_iters = self.batch_size * 100
        self.epochs = 100

        # Data dimensions
        self.convdims = 2  # 2D or 3D convolutions
        self.nz = 1  # Number of channels in Z
        self.zx = 12  # Size of each spatial dimensions in Z
        self.zx_sample = 12
        self.nc = 1  # Number of channels

        # Network setup
        self.discriminator = classical_sgan_disc
        self.generator = classical_sgan_gen
        self.loss_fake = epsilon_gan_fake
        self.loss_true = epsilon_gan_true
        self.loss_gen = epsilon_gan_gen
        self.clip_weights = False  # Clip the discriminator weights (cf Wasserstein GAN)
        self.c = 0.01  # Clipping value
        self.k = 1  # Number of D updates vs G updates

        # Optimizer
        self.optimizer = Adam
        self.lr = 0.001  # learning rate
        self.b1 = 0.5  # Adam momentum term
        self.optimizer_params = {"lr": self.lr, "beta_1": self.b1}


if __name__ == "__main__":
    if len(sys.argv) == 2:
        conf = Config(sys.argv[1])
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)

    else:
        conf = Config(str(datetime.datetime.now()))
        with open(conf.name + ".sgancfg", "wb") as f:
            pickle.dump(conf, f)
