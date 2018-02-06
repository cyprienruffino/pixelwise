

def classical_sgan_disc(npx,
                        convdims=2,
                        disc_depth=5,
                        channels=1,
                        clip_weights=False,
                        clipping_value=0.01):
    from kgan.constraints import Clip
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input,
                              LeakyReLU, GaussianNoise)
    from keras.regularizers import l2
    from keras.initializers import RandomNormal

    conv_kernel = 9
    l2_fac = 1e-5
    strides = 2
    epsilon = 1e-4
    depth=5
    convs = [pow(2,i+6) for i in range(disc_depth - 1)] + [1]
    init = RandomNormal(stddev=0.02)

    # Setup
    if convdims == 2:
        Conv = Conv2D
    elif convdims == 3:
        Conv = Conv3D

    if clip_weights:
        W_constraint = Clip(clipping_value)
    else:
        W_constraint = None

    X = Input((channels, ) + (npx, ) * convdims, name="X")

    # Discriminator
    layer = GaussianNoise(stddev=0.1)(X)
    layer = Conv(
        filters=convs[0],
        kernel_size=conv_kernel,
        padding="same",
        strides=strides,
        use_bias=False,
        kernel_initializer=init,
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
            use_bias=False,
            kernel_initializer=init,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first",
            kernel_constraint=W_constraint)(layer)
        layer = LeakyReLU()(conv)

    D_out = Conv(
        filters=convs[-1],
        kernel_size=conv_kernel,
        activation="sigmoid",
        padding="same",
        strides=strides,
        use_bias=False,
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
        kernel_constraint=W_constraint)(layer)

    return Model(inputs=X, outputs=D_out, name="D")
