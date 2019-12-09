import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl

from layers import InstanceNormalization, ResidualBlock


def create_network(Zt, Ct,
                   channels=1,
                   encoder_filters=[64, 128, 256],
                   encoder_ks=[7, 3, 3],
                   encoder_strides=[1, 2, 2],
                   resblock_filters=[256, 256, 256, 256, 256, 256],
                   resblock_ks=[3, 3, 3, 3, 3, 3],
                   decoder_filters=[128, 64],
                   decoder_ks=[3, 3, 7],
                   decoder_strides=[2, 2, 1]
                   ):
    with tf.name_scope("Gen"):

        Z = kl.Input((None, None, channels,), tensor=Zt, name="Z")
        C = kl.Input((None, None, channels,), tensor=Ct, name="C")

        # Encoder
        layer = C
        for l in range(len(encoder_filters)):
            layer = kl.Conv2D(
                filters=encoder_filters[l],
                kernel_size=encoder_ks[l],
                padding="same",
                activation="relu",
                strides=encoder_strides[l])(layer)
            layer = InstanceNormalization()(layer)

        layer = kl.concatenate([layer, Z])

        # Transformer
        for l in range(len(resblock_filters)):
            layer = ResidualBlock(resblock_filters[l] + channels, nb_layers=3, kernel_size=resblock_ks[l],
                                  normalization="instancenorm")(layer)

        # Decoder
        for l in range(len(decoder_filters)):
            layer = kl.Conv2DTranspose(
                filters=decoder_filters[l],
                kernel_size=decoder_ks[l],
                padding="same",
                strides=decoder_strides[l],
                activation="relu")(layer)
            layer = InstanceNormalization()(layer)

        G_out = kl.Conv2D(
            filters=channels,
            kernel_size=decoder_ks[-1],
            strides=decoder_strides[-1],
            activation="tanh",
            padding="same")(layer)

        model = k.Model(inputs=[Z, C], outputs=G_out, name="G")
    return model
