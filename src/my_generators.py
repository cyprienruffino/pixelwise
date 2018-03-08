

def classical_sgan_gen(
        filter_size=5,
        convdims=2,
        depth=5,
        channels=1,
        init="glorot_uniform",
        l2_fac=1e-5,
        epsilon=1e-4,
        deconvs=[512, 256, 128, 64, 1]):
    from keras.engine import Model
    from keras.layers import (BatchNormalization, Conv2DTranspose,
                              Conv3DTranspose, Input)
    from keras.regularizers import l2

    if convdims == 2:
        ConvTranspose = Conv2DTranspose
    elif convdims == 3:
        ConvTranspose = Conv3DTranspose

    # Generator
    Z = Input((channels, ) + (None, ) * convdims, name="Z")
    layer = Z

    for l in range(len(deconvs) - 1):
        layer = ConvTranspose(
            filters=deconvs[l],
            kernel_size=filter_size,
            padding="same",
            strides=2,
            kernel_initializer=init,
            kernel_regularizer=l2(l2_fac),
            activation="relu",
            data_format="channels_first")(layer)
        layer = BatchNormalization(axis=1, epsilon=epsilon)(layer)

    G_out = ConvTranspose(
        filters=deconvs[-1],
        kernel_size=filter_size,
        padding="same",
        strides=2,
        activation="tanh",
        kernel_initializer=init,
        kernel_regularizer=l2(l2_fac),
        data_format="channels_first")(layer)

    return Model(inputs=Z, outputs=G_out, name="G")
