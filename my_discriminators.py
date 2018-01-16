def classical_sgan_disc(convdims=2,
                        channels=1,
                        clip_weights=False,
                        clipping_value=0.01):
    from kgan.constraints import Clip
<<<<<<< HEAD
    from kgan.layers import LayerNormalization
=======
>>>>>>> 0b5c0d0697be2f2b4d201a251999cd38520fe751
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2D, Conv3D, Input,
                              LeakyReLU, GaussianNoise)
    from keras.regularizers import l2
    from keras.initializers import RandomNormal

    conv_kernel = 9
    l2_fac = 1e-5
    strides = 2
<<<<<<< HEAD
    epsilon = 1e-4
=======
    epsilon=1e-4
>>>>>>> 0b5c0d0697be2f2b4d201a251999cd38520fe751
    convs = [64, 128, 256, 512, 1]
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

    X = Input((channels, ) + (None, ) * convdims, name="X")

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
<<<<<<< HEAD
=======
        layer = BatchNormalization(axis=1, epsilon=epsilon)(layer)
>>>>>>> 0b5c0d0697be2f2b4d201a251999cd38520fe751

    D_out = Conv(
        filters=convs[-1],
        kernel_size=conv_kernel,
        activation="sigmoid",
        padding="same",
        use_bias=False,
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first",
<<<<<<< HEAD
        kernel_constraint=W_constraint)(layer)
=======
        kernel_constraint=W_constraint,
        name="D_out")(layer)
>>>>>>> 0b5c0d0697be2f2b4d201a251999cd38520fe751

    return Model(inputs=X, outputs=D_out, name="D")
