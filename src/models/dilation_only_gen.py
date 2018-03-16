from keras.engine import Model
from keras.initializers import RandomNormal
from keras.layers import (BatchNormalization, Input,
                          Conv2DTranspose)
from keras.layers import concatenate
from keras.regularizers import l2


def create_network(
        filter_size=5,
        convdims=2,
        channels=1,
        l2_fac=1e-5,
        dilations=[2, 2, 2, 2, 2],
        epsilon=1e-4,
        filters=[64, 128, 256, 512, 1],
        init="glorot_uniform"):
    Z = Input((channels,) + (None,) * convdims, name="Z")
    C = Input((channels,) + (None,) * convdims, name="C")

    layer = concatenate([Z, C], axis=1)

    # Generator
    for l in range(len(filters) - 1):
        layer = Conv2DTranspose(
            filters=filters[l],
            kernel_size=filter_size,
            padding="same",
            dilation_rate=dilations[l],
            kernel_initializer=init,
            kernel_regularizer=l2(l2_fac),
            activation="relu",
            data_format="channels_first")(layer)
        layer = BatchNormalization(axis=1, epsilon=epsilon, gamma_initializer=RandomNormal(mean=1, stddev=0.02))(layer)

    G_out = Conv2DTranspose(
        filters=filters[-1],
        kernel_size=filter_size,
        activation="sigmoid",
        padding="same",
        dilation_rate=dilations[-1],
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first")(layer)

    return Model(inputs=[Z, C], outputs=G_out, name="D")
