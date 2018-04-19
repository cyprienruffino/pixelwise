from keras.layers import concatenate, Conv2D, Conv3D
from keras.engine import Model
from keras.layers import (BatchNormalization, Conv2DTranspose,
                          Conv3DTranspose, Input)
from keras.regularizers import l2
from keras.initializers import RandomNormal


def create_network(
        filter_size=5,
        convdims=2,
        channels=1,
        strides=[2, 2, 2, 2, 2],
        dilations=[1, 1, 2, 2, 3, 3],
        init="glorot_uniform",
        l2_fac=1e-5,
        epsilon=1e-4,
        upscaling_filters=[512, 256, 128, 64, 1],
        dilations_filters=[64, 128, 256, 512, 1]):

    # Generator
    Z = Input((channels, ) + (None, ) * convdims, name="Z")
    layer = Z

    if convdims == 2:
        ConvTranspose = Conv2DTranspose
        Conv = Conv2D
    elif convdims == 3:
        ConvTranspose = Conv3DTranspose
        Conv = Conv3D

    for l in range(len(upscaling_filters) - 1):
        layer = ConvTranspose(
            filters=upscaling_filters[l],
            kernel_size=filter_size,
            padding="same",
            strides=strides[l],
            kernel_initializer=init,
            kernel_regularizer=l2(l2_fac),
            activation="relu",
            data_format="channels_first")(layer)
        layer = BatchNormalization(axis=1, epsilon=epsilon, gamma_initializer=RandomNormal(mean=1, stddev=0.02))(layer)

    layer = ConvTranspose(
        filters=upscaling_filters[-1],
        kernel_size=filter_size,
        padding="same",
        strides=strides[-1],
        activation="relu",
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first")(layer)

    # Dilation
    for l in range(len(dilations_filters) - 1):
        layer = Conv(
            filters=dilations_filters[l],
            kernel_size=filter_size,
            padding="same",
            dilation_rate=dilations[l],
            activation="relu",
            kernel_initializer=init,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first")(layer)
        layer = BatchNormalization(axis=1, epsilon=epsilon, gamma_initializer=RandomNormal(mean=1, stddev=0.02))(layer)

    G_out = Conv(
        filters=dilations_filters[-1],
        kernel_size=filter_size,
        activation="tanh",
        padding="same",
        dilation_rate=dilations[-1],
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first")(layer)

    return Model(inputs=Z, outputs=G_out, name="G")
