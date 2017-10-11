from keras.engine import Model
from keras.initializers import Constant, RandomNormal
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Conv3D,
                          Conv3DTranspose, GaussianNoise, Input, LeakyReLU)
from keras.regularizers import l2


def sgan(config):
    # Setup
    if config.convdims == 2:
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
    elif config.convdims == 3:
        Conv = Conv3D
        ConvTranspose = Conv3DTranspose

    weights_init = RandomNormal(stddev=0.02)
    beta_init = Constant(value=0.0)
    gamma_init = RandomNormal(mean=1., stddev=0.02)

    # Inputs
    Z = Input((config.nz, ) + (None, ) * config.convdims, name="Z")
    X = Input((config.nc,) + (config.npx,) * config.convdims), name="X")

    # Generator
    layer = Z
    for l in range(config.gen_depth - 1):
        tconv = ConvTranspose(
            filters=config.gen_fn[l],
            kernel_size=config.gen_ks[l],
            activation="relu",
            kernel_regularizer=l2(config.l2_fac),
            data_format="channels_first",
            kernel_initializer=weights_init)(layer)
        layer = BatchNormalization(
            gamma_initializer=gamma_init, beta_initializer=beta_init,
            axis=1)(tconv)

    G_out = ConvTranspose(
        filters=config.gen_fn[-1],
        kernel_size=config.gen_ks[-1],
        activation="tanh",
        kernel_regularizer=l2(config.l2_fac),
        data_format="channels_first",
        kernel_initializer=weights_init,
        name="G_out")(layer)

    # Discriminator
    noise = GaussianNoise(stddev=0.1)(X)
    layer = Conv(
        filters=config.dis_fn[0],
        kernel_size=config.dis_ks[0],
        activation="linear",
        kernel_regularizer=l2(config.l2_fac),
        data_format="channels_first",
        kernel_initializer=weights_init)(noise)
    layer = LeakyReLU()(layer)

    for l in range(1, config.dis_depth - 1):
        conv = Conv(
            filters=config.dis_fn[l],
            kernel_size=config.dis_ks[l],
            activation="linear",
            kernel_regularizer=l2(config.l2_fac),
            data_format="channels_first",
            kernel_initializer=weights_init)(layer)
        layer = LeakyReLU()(conv)
        layer = BatchNormalization(
            gamma_initializer=gamma_init, beta_initializer=beta_init,
            axis=1)(layer)

    D_out = Conv(
        filters=config.dis_fn[-1],
        kernel_size=config.dis_ks[-1],
        activation="sigmoid",
        kernel_regularizer=l2(config.l2_fac),
        data_format="channels_first",
        kernel_initializer=weights_init,
        name="D_out")(layer)

    # Models
    D = Model(inputs=X, outputs=D_out, name="D")
    G = Model(inputs=Z, outputs=G_out, name="G")
    DG = Model(inputs=Z, outputs=D(G(Z)), name="DG")
    Adv = Model(inputs=[X, Z], outputs=[D(X), D(G(Z))], name="Adv")

    return D, G, DG, Adv
