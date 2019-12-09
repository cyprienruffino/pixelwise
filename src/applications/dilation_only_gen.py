import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import regularizers as kr


def create_network(Zt, Ct,
        filter_size=5,
        channels=1,
        dilations=[2, 2, 2, 2, 2],
        filters=[64, 128, 256, 512, 1]):

    with tf.name_scope("Gen"):

        Z = kl.Input((None, None, channels,), tensor=Zt, name="Z")
        C = kl.Input((None, None, channels,), tensor=Ct, name="C")

        layer = kl.concatenate([Z, C], axis=1)

        # Generator
        for l in range(len(filters) - 1):
            layer = kl.Conv2DTranspose(
                filters=filters[l],
                kernel_size=filter_size,
                padding="same",
                dilation_rate=dilations[l],
                kernel_regularizer=kr.l2(),
                activation="relu")(layer)
            layer = kl.BatchNormalization()(layer)

        G_out = kl.Conv2DTranspose(
            filters=filters[-1],
            kernel_size=filter_size,
            activation="sigmoid",
            padding="same",
            dilation_rate=dilations[-1],
            kernel_regularizer=kr.l2())(layer)

        model = k.Model(inputs=[Z, C], outputs=G_out)
    return model
