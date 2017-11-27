from keras.engine import Model
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Conv3D,
                          Conv3DTranspose, Input, LeakyReLU, UpSampling2D,
                          UpSampling3D)
from config import GenUpscaling
from keras.regularizers import l2


def create_gen(config):
    conv = None
    G_out = None

    # Generator
    if config.convdims == 2:
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
        Upsampling = UpSampling2D
    elif config.convdims == 3:
        Conv = Conv3D
        ConvTranspose = Conv3DTranspose
        Upsampling = UpSampling3D

    Z = Input((config.nz, ) + (None, ) * config.convdims, name="Z")

    if config.gen_up == GenUpscaling.upsampling:
        layer = Upsampling(data_format="channels_first")(Z)
    else:
        layer = Z

    for l in range(config.gen_depth - 1):

        if config.gen_up == GenUpscaling.deconvolution:
            conv = ConvTranspose(
                filters=config.gen_fn[l],
                kernel_size=config.gen_ks[l],
                padding="same",
                strides=config.gen_strides[l],
                kernel_regularizer=l2(config.l2_fac),
                data_format="channels_first")(layer)

        elif config.gen_up == GenUpscaling.upsampling:
            conv = Conv(
                filters=config.gen_fn[l],
                kernel_size=config.gen_ks[l],
                padding="same",
                kernel_regularizer=l2(config.l2_fac),
                data_format="channels_first")(layer)

        layer = LeakyReLU(alpha=0.2)(conv)
        layer = BatchNormalization(axis=1)(layer)

        if config.gen_up == GenUpscaling.upsampling:
            layer = Upsampling(data_format="channels_first")(layer)

    if config.gen_up == GenUpscaling.upsampling:
        G_out = Conv(
            filters=config.gen_fn[-1],
            kernel_size=config.gen_ks[-1],
            activation="tanh",
            padding="same",
            kernel_regularizer=l2(config.l2_fac),
            data_format="channels_first",
            name="G_out")(layer)

    elif config.gen_up == GenUpscaling.deconvolution:
        G_out = ConvTranspose(
            filters=config.gen_fn[-1],
            kernel_size=config.gen_ks[-1],
            activation="tanh",
            padding="same",
            kernel_regularizer=l2(config.l2_fac),
            data_format="channels_first",
            name="G_out")(layer)

    return Model(inputs=Z, outputs=G_out, name="G")
