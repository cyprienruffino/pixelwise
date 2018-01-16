def classical_sgan_gen(convdims=2,
                       channels=1,
                       clip_weights=False,
                       clipping_value=0.01):
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2DTranspose,
                              Conv3DTranspose, Input, LeakyReLU)
    from keras.regularizers import l2
    from keras.initializers import RandomNormal

    if convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif convdims == 3:
        ConvTranspose = Conv3DTranspose

    init = RandomNormal(stddev=0.02)
    conv = None
    G_out = None
    deconv_kernel = 5
    l2_fac = 1e-5
    epsilon = 1e-4
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
            kernel_initializer=init,
            kernel_regularizer=l2(l2_fac),
            data_format="channels_first")(layer)
        layer = LeakyReLU()(conv)
        layer = BatchNormalization(axis=1, epsilon=epsilon)(layer)

    G_out = ConvTranspose(
        filters=deconvs[-1],
        kernel_size=deconv_kernel,
        padding="same",
        strides=2,
        activation="tanh",
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first")(layer)

    return Model(inputs=Z, outputs=G_out, name="G")
