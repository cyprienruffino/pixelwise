import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import regularizers as kr

from layers import InstanceNormalization


def create_network(Xt, Ct,
                   filter_size=4,
                   img_shape=(None, None, 1),
                   strides=[2, 2, 2, 2, 1],
                   filters=[64, 128, 256, 512, 1]):
    with tf.name_scope("Disc"):
        X = kl.Input(img_shape, tensor=Xt, name="X")
        C = kl.Input(img_shape, tensor=Ct, name="C")

        layer = kl.GaussianNoise(stddev=0.1)(X)
        layer = kl.concatenate([layer, C], axis=-1)

        # Discriminator
        for l in range(len(filters) - 1):
            conv = kl.Conv2D(
                filters=filters[l],
                kernel_size=filter_size,
                padding="same",
                strides=strides[l])(layer)
            layer = kl.LeakyReLU()(conv)
            layer = InstanceNormalization()(layer)

        D_out = kl.Conv2D(
            filters=img_shape[-1],
            kernel_size=filter_size,
            activation="sigmoid",
            padding="same",
            strides=strides[-1])(layer)

        model = k.Model(inputs=[X, C], outputs=D_out)
    return model
