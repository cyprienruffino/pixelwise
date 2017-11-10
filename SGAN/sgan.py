import keras.backend as K
from keras.constraints import Constraint
from keras.engine import Model
from keras.initializers import Constant, RandomNormal
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Conv3D,
                          Conv3DTranspose, GaussianNoise, Input, LeakyReLU,
                          UpSampling2D, UpSampling3D)
from keras.optimizers import Adam
from keras.regularizers import l2

from config import GenUpscaling, Losses
from tools import TimePrint


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''

    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__, 'c': self.c}


def sgan(config):
    # Setup
    if config.convdims == 2:
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
        Upsampling = UpSampling2D
    elif config.convdims == 3:
        Conv = Conv3D
        ConvTranspose = Conv3DTranspose
        Upsampling = UpSampling3D

    if config.clip_gradients:
        W_constraint = WeightClip(config.c)
    else:
        W_constraint = None

    weights_init = RandomNormal(stddev=0.02)
    beta_init = Constant(value=0.0)
    gamma_init = RandomNormal(mean=1., stddev=0.02)

    # Inputs
    Z = Input((config.nz, ) + (None, ) * config.convdims, name="Z")
    X = Input((config.nc, ) + (None, ) * config.convdims, name="X")

    conv = None
    G_out = None

    # Generator
    if config.gen_up == GenUpscaling.upsampling:
        layer = Upsampling(data_format="channels_first")(Z)
    else:
        layer = Z

    for l in range(config.gen_depth - 1):

        if config.gen_up == GenUpscaling.deconvolution:
            conv = ConvTranspose(
                filters=config.gen_fn[l],
                kernel_size=config.gen_ks[l],
                activation="linear",
                padding="same",
                strides=config.gen_strides[l],
                kernel_regularizer=l2(config.l2_fac),
                data_format="channels_first",
                kernel_initializer=weights_init)(layer)

        elif config.gen_up == GenUpscaling.upsampling:
            conv = Conv(
                filters=config.gen_fn[l],
                kernel_size=config.gen_ks[l],
                activation="linear",
                padding="same",
                kernel_regularizer=l2(config.l2_fac),
                data_format="channels_first",
                kernel_initializer=weights_init)(layer)

        layer = LeakyReLU(alpha=0.2)(conv)
        layer = BatchNormalization(
            gamma_initializer=gamma_init, beta_initializer=beta_init,
            axis=1)(layer)

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
            kernel_initializer=weights_init,
            name="G_out")(layer)

    elif config.gen_up == GenUpscaling.deconvolution:
        G_out = ConvTranspose(
            filters=config.gen_fn[-1],
            kernel_size=config.gen_ks[-1],
            activation="tanh",
            padding="same",
            kernel_regularizer=l2(config.l2_fac),
            data_format="channels_first",
            kernel_initializer=weights_init,
            name="G_out")(layer)

    # Discriminator
    if config.noise:
        layer = GaussianNoise(stddev=0.1)(X)
    else:
        layer = X
    layer = Conv(
        filters=config.dis_fn[0],
        kernel_size=config.dis_ks[0],
        activation="linear",
        padding="same",
        strides=config.dis_strides[0],
        kernel_regularizer=l2(config.l2_fac),
        data_format="channels_first",
        kernel_initializer=weights_init,
        kernel_constraint=W_constraint)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    for l in range(1, config.dis_depth - 1):
        conv = Conv(
            filters=config.dis_fn[l],
            kernel_size=config.dis_ks[l],
            activation="linear",
            padding="same",
            strides=config.dis_strides[l],
            kernel_regularizer=l2(config.l2_fac),
            data_format="channels_first",
            kernel_initializer=weights_init,
            kernel_constraint=W_constraint)(layer)
        layer = LeakyReLU(alpha=0.2)(conv)
        layer = BatchNormalization(
            gamma_initializer=gamma_init, beta_initializer=beta_init,
            axis=1)(layer)

    D_out = Conv(
        filters=config.dis_fn[-1],
        kernel_size=config.dis_ks[-1],
        activation="sigmoid",
        padding="same",
        kernel_regularizer=l2(config.l2_fac),
        data_format="channels_first",
        kernel_initializer=weights_init,
        kernel_constraint=W_constraint,
        name="D_out")(layer)

    # Selecting the losses
    if config.losses == Losses.classical_gan:
        from losses import gan_true as loss_true
        from losses import gan_fake as loss_fake
        from losses import gan_gen as loss_gen
    elif config.losses == Losses.epsilon_gan:
        from losses import epsilon_gan_true as loss_true
        from losses import epsilon_gan_fake as loss_fake
        from losses import epsilon_gan_gen as loss_gen
    elif config.losses == Losses.wasserstein_gan:
        from losses import wasserstein_true as loss_true
        from losses import wasserstein_fake as loss_fake
        from losses import wasserstein_gen as loss_gen
    elif config.losses == Losses.softplus_gan:
        from losses import softplus_gan_true as loss_true
        from losses import softplus_gan_fake as loss_fake
        from losses import softplus_gan_gen as loss_gen
    else:
        raise "Unknown losses"

    # Creating and compiling the models
    TimePrint("Compiling the network...\n")

    D = Model(inputs=X, outputs=D_out, name="D")
    G = Model(inputs=Z, outputs=G_out, name="G")

    for layer in G.layers:
        layer.trainable = False
    Adv = Model(inputs=[X, Z], outputs=[D(X), D(G(Z))], name="Adv")
    # Keras sums the losses
    Adv.compile(
        optimizer=Adam(lr=config.lr, beta_1=config.b1),
        loss=[loss_true, loss_fake],
        loss_weights=[1, 1])
    TimePrint("Discriminator done.")

    for layer in D.layers:
        layer.trainable = False
    for layer in G.layers:
        layer.trainable = True
    DG = Model(inputs=Z, outputs=D(G(Z)), name="DG")
    DG.compile(optimizer=Adam(lr=config.lr, beta_1=config.b1), loss=loss_gen)
    TimePrint("Generator done.")
    for layer in D.layers:
        layer.trainable = True

    return D, G, DG, Adv
