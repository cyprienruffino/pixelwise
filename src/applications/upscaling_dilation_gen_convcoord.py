import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import regularizers as kr
from layers.convcoord import ConvCoord2D


def create_network(Zt, Ct,
                   filter_size=5,
                   strides=[2, 2, 2, 2, 1],
                   dilations=[1, 2, 3, 4, 5],
                   img_shape=(None, None, 1),
                   noise_shape=(None, None, 1),
                   upscaling_filters=[512, 256, 128, 64, 32],
                   dilations_filters=[64, 128, 256, 512]):

    with tf.name_scope("Gen"):

        # Generator
        Z = kl.Input(noise_shape, tensor=Zt, name="Z")
        C = kl.Input(img_shape, tensor=Ct, name="C")
        layer = ConvCoord2D()(Z)
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
            strides=strides[-1],
            padding="same",
            activation="relu",
            kernel_regularizer=kr.l2())(layer)

        layer = kl.concatenate([layer, C])
        layer = ConvCoord2D()(layer)

        # Dilation
        for l in range(len(dilations_filters) - 1):
            layer = kl.Conv2D(
                filters=dilations_filters[l],
                kernel_size=filter_size,
                padding="same",
                dilation_rate=dilations[l],
                activation="relu",
                kernel_regularizer=kr.l2())(layer)
            layer = kl.BatchNormalization()(layer)

        G_out = kl.Conv2D(
            filters=img_shape[-1],
            kernel_size=filter_size,
            activation="tanh",
            padding="same",
            dilation_rate=dilations[-1],
            kernel_regularizer=kr.l2())(layer)

        model = k.Model(inputs=[Z, C], outputs=G_out, name="G")
    return model
