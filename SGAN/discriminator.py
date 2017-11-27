import keras.backend as K
from keras.engine import Model
from keras.constraints import Constraint
from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input, LeakyReLU,
                          GaussianNoise)
from keras.regularizers import l2


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''

    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__, 'c': self.c}


def create_disc(config):

    # Setup
    if config.convdims == 2:
        Conv = Conv2D
    elif config.convdims == 3:
        Conv = Conv3D

    if config.clip_gradients:
        W_constraint = WeightClip(config.c)
    else:
        W_constraint = None

    X = Input((config.nc, ) + (None, ) * config.convdims, name="X")

    # Discriminator
    if config.noise:
        layer = GaussianNoise(stddev=0.1)(X)
    else:
        layer = X
    layer = Conv(
        filters=config.dis_fn[0],
        kernel_size=config.dis_ks[0],
        padding="same",
        strides=config.dis_strides[0],
        kernel_regularizer=l2(config.l2_fac),
        data_format="channels_first",
        kernel_constraint=W_constraint)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    for l in range(1, config.dis_depth - 1):
        conv = Conv(
            filters=config.dis_fn[l],
            kernel_size=config.dis_ks[l],
            padding="same",
            strides=config.dis_strides[l],
            kernel_regularizer=l2(config.l2_fac),
            data_format="channels_first",
            kernel_constraint=W_constraint)(layer)
        layer = LeakyReLU(alpha=0.2)(conv)
        layer = BatchNormalization(axis=1)(layer)

    D_out = Conv(
        filters=config.dis_fn[-1],
        kernel_size=config.dis_ks[-1],
        activation="sigmoid",
        padding="same",
        kernel_regularizer=l2(config.l2_fac),
        data_format="channels_first",
        kernel_constraint=W_constraint,
        name="D_out")(layer)

    return Model(inputs=X, outputs=D_out, name="D")
