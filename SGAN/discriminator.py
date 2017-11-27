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
    conv_kernel = 3
    l2_fac = 1e-5
    strides = 2
    convs = [128, 256, 512]

    # Setup
    if config.convdims == 2:
        Conv = Conv2D
    elif config.convdims == 3:
        Conv = Conv3D

    if config.clip_weights:
        W_constraint = WeightClip(config.c)
    else:
        W_constraint = None

    X = Input((config.nc, ) + (None, ) * config.convdims, name="X")

    # Discriminator
    layer = GaussianNoise(stddev=0.1)(X)
    layer = Conv(
        filters=convs[0],
        kernel_size=conv_kernel,
        padding="same",
        strides=strides,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
        kernel_constraint=W_constraint)(layer)
    layer = LeakyReLU()(layer)

    for l in range(1, len(convs) - 1):
        conv = Conv(
            filters=convs[l],
            kernel_size=conv_kernel,
            padding="same",
            strides=strides,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first",
            kernel_constraint=W_constraint)(layer)
        layer = LeakyReLU()(conv)
        layer = BatchNormalization(axis=1)(layer)

    D_out = Conv(
        filters=convs[-1],
        kernel_size=conv_kernel,
        activation="sigmoid",
        padding="same",
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
        kernel_constraint=W_constraint,
        name="D_out")(layer)

    return Model(inputs=X, outputs=D_out, name="D")
