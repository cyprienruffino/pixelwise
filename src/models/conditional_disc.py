from keras.layers import concatenate, Conv2D
from keras.engine import Model
from keras.layers import (BatchNormalization, Input,
                          LeakyReLU, GaussianNoise)
from keras.initializers import RandomNormal
from keras.regularizers import l2

def create_network(
        filter_size=5,
        convdims=2,
        channels=1,
        l2_fac=1e-5,
        gaussian_noise_stddev=0.1,
        strides=[2, 2, 2, 2, 2],
        alpha=0.2,
        epsilon=1e-4,
        filters=[64, 128, 256, 512, 1],
        init="glorot_uniform"):


    X = Input((channels,) + (None,) * convdims, name="X")
    C = Input((channels,) + (None,) * convdims, name="C")

    layer = concatenate([X, C], axis=1)

    # Discriminator
    layer = GaussianNoise(stddev=gaussian_noise_stddev)(layer)
    layer = Conv2D(
        filters=filters[0],
        kernel_size=filter_size,
        padding="same",
        strides=2,
        use_bias=False,
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first")(layer)
    layer = LeakyReLU(alpha)(layer)

    for l in range(1, len(filters) - 1):
        conv = Conv2D(
            filters=filters[l],
            kernel_size=filter_size,
            padding="same",
            strides=strides[l],
            use_bias=False,
            kernel_initializer=init,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first")(layer)
        layer = LeakyReLU(alpha)(conv)
        layer = BatchNormalization(axis=1, epsilon=epsilon, gamma_initializer=RandomNormal(mean=1, stddev=0.02))(layer)

    D_out = Conv2D(
        filters=filters[-1],
        kernel_size=filter_size,
        activation="sigmoid",
        padding="same",
        strides=strides[-1],
        use_bias=False,
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first")(layer)

    return Model(inputs=[X, C], outputs=D_out, name="D")
