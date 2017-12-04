from keras.engine import Model
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Conv3D,
                          Activation, Conv3DTranspose, Input, LeakyReLU,
                          concatenate)
from keras.regularizers import l2


def create_gen(config):
    return _classical_sgan(config)


def _classical_sgan(config):
    if config.convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif config.convdims == 3:
        ConvTranspose = Conv3DTranspose

    conv = None
    G_out = None

    deconv_kernel = 5
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
        activation="tanh",
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
        name="G_out")(layer)

    return Model(inputs=Z, outputs=G_out, name="G")


def _pix2pix_decoder(config):
    if config.convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif config.convdims == 3:
        ConvTranspose = Conv3DTranspose

    conv = None
    G_out = None

    deconv_kernel = 4
    l2_fac = 1e-5

    decoder = [512, 512, 512, 512, 512, 256, 128, 64, 1]

    # Generator
    Z = Input((config.nz, ) + (None, ) * config.convdims, name="Z")
    layer = Z

    # Decoder
    for l in range(len(decoder) - 1):
        conv = ConvTranspose(
            filters=decoder[l],
            kernel_size=deconv_kernel,
            padding="same",
            strides=2,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first")(layer)
        layer = BatchNormalization(axis=1)(conv)
        layer = Activation("relu")(layer)
        layer = Dropout(0.5)(layer)

    G_out = ConvTranspose(
        filters=decoder[-1],
        kernel_size=deconv_kernel,
        padding="same",
        strides=2,
        activation="tanh",
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
        name="G_out")(layer)

    return Model(inputs=Z, outputs=G_out, name="G")


def _pix2pix_encoder_decoder(config):
    if config.convdims == 2:
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
    elif config.convdims == 3:
        Conv = Conv3D
        ConvTranspose = Conv3DTranspose

    conv = None
    G_out = None

    deconv_kernel = 4
    l2_fac = 1e-5

    encoder = [64, 128, 256, 512, 512, 512, 512, 512]
    decoder = [512, 512, 512, 512, 512, 256, 128, 64, 1]

    # Generator
    Z = Input((config.nz, ) + (None, ) * config.convdims, name="Z")
    layer = Z

    # Encoder
    for l in range(len(encoder)):
        conv = Conv(
            filters=encoder[l],
            kernel_size=deconv_kernel,
            padding="same",
            strides=2,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first")(layer)
        layer = BatchNormalization(axis=1)(conv)
        layer = Activation("relu")(layer)

    # Decoder
    for l in range(len(decoder) - 1):
        conv = ConvTranspose(
            filters=decoder[l],
            kernel_size=deconv_kernel,
            padding="same",
            strides=2,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first")(layer)
        layer = BatchNormalization(axis=1)(conv)
        layer = Activation("relu")(layer)
        layer = Dropout(0.5)(layer)

    G_out = ConvTranspose(
        filters=decoder[-1],
        kernel_size=deconv_kernel,
        padding="same",
        strides=2,
        activation="tanh",
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
        name="G_out")(layer)

    return Model(inputs=Z, outputs=G_out, name="G")


def _pix2pix_unet(config):
    if config.convdims == 2:
        ConvTranspose = Conv2DTranspose
        Conv = Conv2D
    elif config.convdims == 3:
        ConvTranspose = Conv3DTranspose
        Conv = Conv2D

    conv = None
    G_out = None

    deconv_kernel = 4
    l2_fac = 1e-5
    encoder = [64, 128, 256, 512, 512, 512, 512, 512]
    decoder = [512, 512, 512, 512, 512, 256, 128, 64, 1]

    # Generator
    Z = Input((config.nz, ) + (None, ) * config.convdims, name="Z")
    layer = Z

    # Encoder
    convs = []
    for l in range(len(encoder)):
        conv = Conv(
            filters=encoder[l],
            kernel_size=deconv_kernel,
            padding="same",
            strides=2,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first")(layer)
        convs.append(conv)
        layer = Activation("relu")(layer)
        layer = BatchNormalization(axis=1)(layer)

    # Decoder
    for l in range(len(decoder) - 1):
        concat = concatenate([layer, convs[len(encoder) - l - 1]], axis=1)
        conv = ConvTranspose(
            filters=decoder[l],
            kernel_size=deconv_kernel,
            padding="same",
            strides=2,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first")(concatenate)
        layer = Activation("relu")(layer)
        layer = BatchNormalization(axis=1)(layer)

    G_out = ConvTranspose(
        filters=decoder[-1],
        kernel_size=deconv_kernel,
        padding="same",
        strides=2,
        activation="tanh",
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
        name="G_out")(layer)

    return Model(inputs=Z, outputs=G_out, name="G")


def _pix2pix_patchgan(config):
    if config.convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif config.convdims == 3:
        ConvTranspose = Conv3DTranspose

    raise NotImplemented
