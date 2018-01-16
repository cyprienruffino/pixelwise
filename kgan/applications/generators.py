def sgan4_gen(convdims=2, channels=1, clip_weights=False, clipping_value=0.01):
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2DTranspose,
                              Conv3DTranspose, Input, LeakyReLU)
    from keras.regularizers import l2
    if convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif convdims == 3:
        ConvTranspose = Conv3DTranspose

    conv = None
    G_out = None

    deconv_kernel = 5
    l2_fac = 1e-5
    deconvs = [256, 128, 64, 1]

    # Generator
    Z = Input((channels, ) + (None, ) * convdims, name="Z")
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


def sgan5_gen(convdims=2, channels=1, clip_weights=False, clipping_value=0.01):
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2DTranspose,
                              Conv3DTranspose, Input, LeakyReLU)
    from keras.regularizers import l2
    if convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif convdims == 3:
        ConvTranspose = Conv3DTranspose

    conv = None
    G_out = None

    deconv_kernel = 5
    l2_fac = 1e-5
    deconvs = [512, 256, 128, 64, 1]

    # Generator
    Z = Input((channels, ) + (None, ) * convdims, name="Z")
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


def sgan6_gen(convdims=2, channels=1, clip_weights=False, clipping_value=0.01):
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2DTranspose,
                              Conv3DTranspose, Input, LeakyReLU)
    from keras.regularizers import l2
    if convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif convdims == 3:
        ConvTranspose = Conv3DTranspose

    conv = None
    G_out = None

    deconv_kernel = 5
    l2_fac = 1e-5
    deconvs = [1024, 512, 256, 128, 64, 1]

    # Generator
    Z = Input((channels, ) + (None, ) * convdims, name="Z")
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


def pix2pix_decoder_gen(convdims=2,
                        channels=1,
                        clip_weights=False,
                        clipping_value=0.01):
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2DTranspose,
                              Conv3DTranspose, Input, Activation, Dropout)
    if convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif convdims == 3:
        ConvTranspose = Conv3DTranspose

    conv = None
    G_out = None

    deconv_kernel = 4
    l2_fac = 1e-5

    decoder = [512, 512, 512, 512, 512, 256, 128, 64, 1]

    # Generator
    Z = Input((channels, ) + (None, ) * convdims, name="Z")
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


def pix2pix_encoder_decoder_gen(convdims=2,
                                channels=1,
                                clip_weights=False,
                                clipping_value=0.01):
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2DTranspose,
                              Conv3DTranspose, Input, Activation, Conv2D,
                              Conv3D, Dropout)
    if convdims == 2:
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
    elif convdims == 3:
        Conv = Conv3D
        ConvTranspose = Conv3DTranspose

    conv = None
    G_out = None

    deconv_kernel = 4
    l2_fac = 1e-5

    encoder = [64, 128, 256, 512, 512, 512, 512, 512]
    decoder = [512, 512, 512, 512, 512, 256, 128, 64, 1]

    # Generator
    Z = Input((channels, ) + (None, ) * convdims, name="Z")
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


def _pix2pix_unet_gen(convdims=2,
                      channels=1,
                      clip_weights=False,
                      clipping_value=0.01):
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2DTranspose,
                              Conv3DTranspose, Input, Activation, concatenate,
                              Conv2D, Conv3D, Dropout)
    if convdims == 2:
        ConvTranspose = Conv2DTranspose
        Conv = Conv2D
    elif convdims == 3:
        ConvTranspose = Conv3DTranspose
        Conv = Conv2D

    conv = None
    G_out = None

    deconv_kernel = 4
    l2_fac = 1e-5
    encoder = [64, 128, 256, 512, 512, 512, 512, 512]
    decoder = [512, 512, 512, 512, 512, 256, 128, 64, 1]

    # Generator
    Z = Input((channels, ) + (None, ) * convdims, name="Z")
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
            data_format="channels_first")(concat)
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


def pix2pix_patchgan_gen(convdims=2,
                         channels=1,
                         clip_weights=False,
                         clipping_value=0.01):
    if convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif convdims == 3:
        ConvTranspose = Conv3DTranspose

    raise NotImplemented
