

def create_network(
        filter_size=5,
        convdims=2,
        channels=1,
        clip_weights=False,
        clipping_value=0.01,
        l2_fac=1e-5,
        strides=2,
        epsilon=1e-4,
        filters=[64, 128, 256, 512, 1],
        init="glorot_uniform"):

    from kgan.constraints import Clip
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input,
                              LeakyReLU, GaussianNoise)
    from keras.regularizers import l2

    # Setup
    if convdims == 2:
        Conv = Conv2D
    elif convdims == 3:
        Conv = Conv3D

    if clip_weights:
        W_constraint = Clip(clipping_value)
    else:
        W_constraint = None

    X = Input((channels,) + (None,) * convdims, name="X")

    # Discriminator
    layer = GaussianNoise(stddev=0.1)(X)
    layer = Conv(
        filters=filters[0],
        kernel_size=filter_size,
        padding="same",
        strides=strides,
        use_bias=False,
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
        kernel_constraint=W_constraint)(layer)
    layer = LeakyReLU()(layer)

    for l in range(1, len(filters) - 1):
        conv = Conv(
            filters=filters[l],
            kernel_size=filter_size,
            padding="same",
            strides=strides,
            use_bias=False,
            kernel_initializer=init,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first",
            kernel_constraint=W_constraint)(layer)
        layer = LeakyReLU()(conv)
        layer = BatchNormalization(axis=1, epsilon=epsilon)(layer)

    D_out = Conv(
        filters=filters[-1],
        kernel_size=filter_size,
        activation="sigmoid",
        padding="same",
        strides=strides,
        use_bias=False,
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
        kernel_constraint=W_constraint)(layer)

    return Model(inputs=X, outputs=D_out, name="D")
