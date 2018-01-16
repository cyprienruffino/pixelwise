def sgan4_disc(convdims=2, channels=1, clip_weights=False,
               clipping_value=0.01):
    from kgan.constraints import Clip
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input,
                              LeakyReLU, GaussianNoise)
    from keras.regularizers import l2

    conv_kernel = 5
    l2_fac = 1e-5
    strides = 2
    convs = [64, 128, 256, 1]

    # Setup
    if convdims == 2:
        Conv = Conv2D
    elif convdims == 3:
        Conv = Conv3D

    if clip_weights:
        W_constraint = Clip(clipping_value)
    else:
        W_constraint = None

    X = Input((channels, ) + (None, ) * convdims, name="X")

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


def sgan5_disc(convdims=2, channels=1, clip_weights=False,
               clipping_value=0.01):
    from kgan.constraints import Clip
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input,
                              LeakyReLU, GaussianNoise)
    from keras.regularizers import l2

    conv_kernel = 5
    l2_fac = 1e-5
    strides = 2
    convs = [64, 128, 256, 512, 1]

    # Setup
    if convdims == 2:
        Conv = Conv2D
    elif convdims == 3:
        Conv = Conv3D

    if clip_weights:
        W_constraint = Clip(clipping_value)
    else:
        W_constraint = None

    X = Input((channels, ) + (None, ) * convdims, name="X")

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


def sgan6_disc(convdims=2, channels=1, clip_weights=False,
               clipping_value=0.01):
    from kgan.constraints import Clip
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input,
                              LeakyReLU, GaussianNoise)
    from keras.regularizers import l2

    conv_kernel = 5
    l2_fac = 1e-5
    strides = 2
    convs = [64, 128, 256, 512, 1024, 1]

    # Setup
    if convdims == 2:
        Conv = Conv2D
    elif convdims == 3:
        Conv = Conv3D

    if clip_weights:
        W_constraint = Clip(clipping_value)
    else:
        W_constraint = None

    X = Input((channels, ) + (None, ) * convdims, name="X")

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


def pix2pix_70x70_disc(convdims=2,
                       channels=1,
                       clip_weights=False,
                       clipping_value=0.01):
    from kgan.constraints import Clip
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input,
                              LeakyReLU, GaussianNoise)
    from keras.regularizers import l2

    conv_kernel = 4
    l2_fac = 1e-5
    strides = 2
    convs = [64, 128, 256, 512, 1]

    # Setup
    if convdims == 2:
        Conv = Conv2D
    elif convdims == 3:
        Conv = Conv3D

    if clip_weights:
        W_constraint = Clip(clipping_value)
    else:
        W_constraint = None

    X = Input((channels, ) + (None, ) * convdims, name="X")

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


def pix2pix_1x1_disc(convdims=2,
                     channels=1,
                     clip_weights=False,
                     clipping_value=0.01):
    from kgan.constraints import Clip
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input,
                              LeakyReLU, GaussianNoise)
    from keras.regularizers import l2

    conv_kernel = 1
    l2_fac = 1e-5
    strides = 2
    convs = [64, 128, 1]

    # Setup
    if convdims == 2:
        Conv = Conv2D
    elif convdims == 3:
        Conv = Conv3D

    if clip_weights:
        W_constraint = Clip(clipping_value)
    else:
        W_constraint = None

    X = Input((channels, ) + (None, ) * convdims, name="X")

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


def pix2pix_16x16_disc(convdims=2,
                       channels=1,
                       clip_weights=False,
                       clipping_value=0.01):
    from kgan.constraints import Clip
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input,
                              LeakyReLU, GaussianNoise)
    from keras.regularizers import l2

    conv_kernel = 4
    l2_fac = 1e-5
    strides = 2
    convs = [64, 128, 1]

    # Setup
    if convdims == 2:
        Conv = Conv2D
    elif convdims == 3:
        Conv = Conv3D

    if clip_weights:
        W_constraint = Clip(clipping_value)
    else:
        W_constraint = None

    X = Input((channels, ) + (None, ) * convdims, name="X")

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


def pix2pix_256x256_disc(convdims=2,
                         channels=1,
                         clip_weights=False,
                         clipping_value=0.01):
    from kgan.constraints import Clip
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input,
                              LeakyReLU, GaussianNoise)
    from keras.regularizers import l2

    conv_kernel = 4
    l2_fac = 1e-5
    strides = 2
    convs = [64, 128, 256, 512, 512, 512, 1]

    # Setup
    if convdims == 2:
        Conv = Conv2D
    elif convdims == 3:
        Conv = Conv3D

    if clip_weights:
        W_constraint = Clip(clipping_value)
    else:
        W_constraint = None

    X = Input((channels, ) + (None, ) * convdims, name="X")

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
