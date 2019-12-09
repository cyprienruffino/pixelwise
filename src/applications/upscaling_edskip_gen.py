import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import regularizers as kr

from layers.residual import ResidualBlock


def create_network(Zt, Ct,
        filter_size=5,
        channels=1,
        strides=[2, 2, 2, 2, 2],
        upscaling_filters=[512, 256, 128, 64, 1],
        encoder_filters=[64, 128, 256],
        decoder_filters=[256, 128, 64],
        resblock_filters=[256, 256, 256, 256, 256],
        resblock_ks=[3, 3, 3, 3, 3, 3]
):

    with tf.name_scope("Gen"):

        # Generator
        Z = kl.Input((None, None, channels,), tensor=Zt, name="Z")
        C = kl.Input((None, None, channels,), tensor=Ct, name="C")
        layer = Z

        # Upscaling
        for l in range(len(upscaling_filters) - 1):
            layer = kl.Conv2DTranspose(
            filters=upscaling_filters[l],
            kernel_size=filter_size,
            padding="same",
            strides=strides[l],
            kernel_regularizer=kr.l2(),
            activation="relu")(layer)
            layer = kl.BatchNormalization()(layer)

        layer = kl.Conv2DTranspose(
        filters=upscaling_filters[-1],
        kernel_size=filter_size,
        padding="same",
        strides=strides[-1],
        activation="relu",
        kernel_regularizer=kr.l2())(layer)

        layer = kl.concatenate([layer, C])

        # Encoder
        skips = []
        for l in range(len(encoder_filters)):
            layer = kl.Conv2D(
            filters=encoder_filters[l],
            kernel_size=filter_size,
            padding="same",
            activation="relu",
            kernel_regularizer=kr.l2())(layer)
            layer = kl.AveragePooling2D()(layer)
            layer = kl.BatchNormalization()(layer)
            skips.append(layer)

        # Residual blocks
        for l in range(len(resblock_filters)):
            layer = ResidualBlock(resblock_filters[l], nb_layers=3, kernel_size=resblock_ks[l])(layer)

        # Decoder
        skips = skips[::-1]
        for l in range(len(decoder_filters)):
            layer = kl.concatenate([layer, skips[l]])
            layer = kl.Conv2DTranspose(
            filters=decoder_filters[l],
            kernel_size=filter_size,
            padding="same",
            strides=strides[l],
            kernel_regularizer=kr.l2(),
            activation="relu")(layer)
            layer = kl.BatchNormalization()(layer)

        G_out = kl.Conv2D(
        filters=1,
        kernel_size=filter_size,
        activation="tanh",
        padding="same",
        kernel_regularizer=kr.l2())(layer)

        model = k.Model(inputs=[Z, C], outputs=G_out, name="G")
    return model
