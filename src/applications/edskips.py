import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import regularizers as kr

from src.layers.residual import ResidualBlock


def create_network(Zt, Ct,
                   channels=1,
                   upscaling_filters=[512, 256, 256, 128, 64],
                   upscaling_ks=[5, 5, 5, 5, 5],
                   upscaling_strides=[2, 2, 2, 2, 2, 1],
                   encoder_filters=[64, 128, 256],
                   encoder_ks=[3, 3, 3],
                   resblock_filters=[256, 256, 256, 256, 256],
                   resblock_ks=[3, 3, 3, 3, 3, 3],
                   decoder_filters=[256, 256, 128, 64],
                   decoder_ks=[5, 5, 5, 5, 5],
                   ):
    with tf.name_scope("Gen"):

        Z = kl.Input((None, None, channels,), tensor=Zt, name="Z")
        C = kl.Input((None, None, channels,), tensor=Ct, name="C")
        layer = Z

        # Upscaling
        for l in range(len(upscaling_filters)):
            layer = kl.Conv2DTranspose(
                filters=upscaling_filters[l],
                kernel_size=upscaling_ks[l],
                padding="same",
                strides=upscaling_strides[l],
                kernel_regularizer=kr.l2(),
                activation="relu")(layer)
            layer = kl.BatchNormalization()(layer)

        layer = kl.Conv2DTranspose(
            filters=channels,
            kernel_size=upscaling_ks[-1],
            padding="same",
            strides=1,
            activation="relu",
            kernel_regularizer=kr.l2())(layer)

        layer = kl.concatenate([layer, C])

        # Encoder
        skips = [C]
        for l in range(len(encoder_filters)):
            layer = kl.Conv2D(
                filters=encoder_filters[l],
                kernel_size=encoder_ks[l],
                padding="same",
                activation="relu",
                # strides=2,
                kernel_regularizer=kr.l2())(layer)
            layer = kl.AveragePooling2D()(layer)
            layer = kl.BatchNormalization()(layer)
            skips.append(layer)

        # Residual blocks
        for l in range(len(resblock_filters)):
            layer = ResidualBlock(resblock_filters[l], nb_layers=3, kernel_size=resblock_ks[l])(layer)

        # Decoder
        layer = kl.Conv2DTranspose(
            filters=decoder_filters[0],
            kernel_size=decoder_ks[0],
            padding="same",
            strides=2,
            kernel_regularizer=kr.l2(),
            activation="relu")(layer)
        layer = kl.BatchNormalization()(layer)

        skips = skips[::-1]
        for l in range(1, len(decoder_filters)):
            layer = kl.concatenate([layer, skips[l]])
            layer = kl.Conv2DTranspose(
                filters=decoder_filters[l],
                kernel_size=decoder_ks[l],
                padding="same",
                strides=2,
                kernel_regularizer=kr.l2(),
                activation="relu")(layer)
            layer = kl.BatchNormalization()(layer)

        G_out = kl.Conv2D(
            filters=channels,
            kernel_size=5,
            activation="tanh",
            padding="same",
            kernel_regularizer=kr.l2())(layer)

        model = k.Model(inputs=[Z, C], outputs=G_out, name="G")
    return model
