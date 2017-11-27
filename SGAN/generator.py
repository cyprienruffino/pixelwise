from keras.engine import Model
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Conv3D,
                          Conv3DTranspose, Input, LeakyReLU)
from keras.regularizers import l2


def create_gen(config):
    conv = None
    G_out = None

    deconv_kernel = 3
    conv_kernel = 3
    l2_fac = 1e-5
    deconvs = [64, 128, 256, 512, 1024]
    convs = [64, 128, 256, 512, 1024]

    if config.convdims == 2:
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
    elif config.convdims == 3:
        Conv = Conv3D
        ConvTranspose = Conv3DTranspose

    # Generator
    Z = Input((config.nz, ) + (None, ) * config.convdims, name="Z")
    layer = Z

    for l in range(len(deconvs) - 1):
        conv = ConvTranspose(
            filters=deconvs[l],
            kernel_size=deconv_kernel,
            padding="same",
            strides=2,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first")(layer)
        layer = LeakyReLU()(conv)
        layer = BatchNormalization(axis=1)(layer)

        conv = Conv(
            filters=convs[l],
            kernel_size=conv_kernel,
            padding="same",
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first")(layer)
        layer = LeakyReLU()(conv)
        layer = BatchNormalization(axis=1)(layer)

    conv = ConvTranspose(
        filters=deconvs[-1],
        kernel_size=deconv_kernel,
        padding="same",
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first")(layer)

    G_out = Conv(
        filters=convs[-1],
        kernel_size=conv_kernel,
        padding="same",
        activation="tanh",
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first")(layer)

    return Model(inputs=Z, outputs=G_out, name="G")
