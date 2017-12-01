from keras.engine import Model
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Conv3D,
                          Conv3DTranspose, Input, LeakyReLU)
from keras.regularizers import l2


def create_gen(config):

    if config.convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif config.convdims == 3:
        ConvTranspose = Conv3DTranspose

    conv = None
    G_out = None

    deconv_kernel = 3
    l2_fac = 1e-5
    deconvs = [512, 256, 128, 64, 1]

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

    G_out = ConvTranspose(
        filters=deconvs[-1],
        kernel_size=deconv_kernel,
        padding="same",
        strides=2,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
        name="G_out")(layer)

    return Model(inputs=Z, outputs=G_out, name="G")
