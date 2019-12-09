import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import regularizers as kr


def create_gen(Zt, Ct,
               img_shape=(28, 28, 1),
               noise_shape=(7, 7, 1),
               filter_size=3,
               strides=[2, 2],
               filters=[128, 64]):

    with tf.name_scope("Gen"):

        # Generator
        Z = kl.Input(noise_shape, tensor=Zt, name="Z")
        C = kl.Input(img_shape, tensor=Ct, name="C")

        Zf = kl.Flatten()(Z)
        layer = kl.Dense(np.prod(noise_shape) * 7)(Zf)
        layer = kl.Reshape(
            (noise_shape[0], noise_shape[1], noise_shape[-1] * 7))(layer)

        for l in range(len(filters)):
            layer = kl.Conv2DTranspose(
                filters=filters[l],
                kernel_size=filter_size,
                padding="same",
                strides=strides[l],
                activation="relu")(layer)
            layer = kl.BatchNormalization()(layer)

        layer = kl.concatenate([layer, C])

        for l in range(len(filters)):
            layer = kl.Conv2DTranspose(
                filters=filters[l],
                kernel_size=filter_size,
                padding="same",
                dilation_rate=l+2,
                activation="relu")(layer)
            layer = kl.BatchNormalization()(layer)

        G_out = kl.Conv2D(
            filters=img_shape[-1],
            kernel_size=filter_size,
            activation="tanh",
            padding="same")(layer)

        model = k.Model(inputs=[Z, C], outputs=G_out)
    return model


def create_disc(Xt, Ct,
                img_shape=(28, 28, 1),
                filter_size=3,
                strides=[2, 2],
                filters=[64, 128]):

    with tf.name_scope("Disc"):
        X = kl.Input(img_shape, tensor=Xt, name="X")
        C = kl.Input(img_shape, tensor=Ct, name="C")

        layer = kl.concatenate([X, C], axis=1)
        layer = kl.GaussianNoise(stddev=0.1)(layer)
        # Discriminator

        layer = kl.Conv2D(
            filters=filters[0],
            kernel_size=filter_size,
            padding="same",
            strides=2)(layer)
        layer = kl.LeakyReLU()(layer)

        for l in range(1, len(filters)):
            conv = kl.Conv2D(
                filters=filters[l],
                kernel_size=filter_size,
                padding="same",
                strides=strides[l])(layer)
            layer = kl.LeakyReLU()(conv)
            layer = kl.Dropout(0.2)(layer)
            layer = kl.BatchNormalization()(layer)

        layer = kl.Flatten()(layer)
        D_out = kl.Dense(1, activation="sigmoid")(layer)

        model = k.Model(inputs=[X, C], outputs=D_out)
    return model


def create_disc_noC(Xt, Ct,
                    img_shape=(28, 28, 1),
                    filter_size=3,
                    strides=[2, 2],
                    filters=[64, 128]):

    with tf.name_scope("Disc"):
        X = kl.Input(img_shape, tensor=Xt, name="X")
        C = kl.Input(img_shape, tensor=Ct, name="C")

        layer = kl.GaussianNoise(stddev=0.1)(X)

        # Discriminator
        layer = kl.Conv2D(
            filters=filters[0],
            kernel_size=filter_size,
            padding="same",
            strides=2)(layer)
        layer = kl.LeakyReLU()(layer)

        for l in range(1, len(filters)):
            conv = kl.Conv2D(
                filters=filters[l],
                kernel_size=filter_size,
                padding="same",
                strides=strides[l])(layer)
            layer = kl.LeakyReLU()(conv)
            layer = kl.BatchNormalization()(layer)

        layer = kl.Flatten()(layer)
        D_out = kl.Dense(1, activation="sigmoid")(layer)

        model = k.Model(inputs=[X, C], outputs=D_out)
    return model
