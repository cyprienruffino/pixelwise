import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import regularizers as kr


def create_classifier():

    with tf.name_scope("Disc"):
        X = kl.Input((28, 28, 1), name="X")
        layer = X


        for l in range(3):
            layer = kl.Conv2D(
                filters=64*(2**l),
                kernel_size=3,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_regularizer=kr.l2())(layer)
            layer = kl.Conv2D(
                filters=64 * (2 ** l),
                kernel_size=3,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_regularizer=kr.l2())(layer)
            layer = kl.MaxPool2D()(layer)
            layer = kl.BatchNormalization()(layer)

        layer = kl.Flatten()(layer)
        layer = kl.Dense(256, kernel_regularizer=kr.l2())(layer)
        layer = kl.LeakyReLU()(layer)
        D_out = kl.Dense(10, activation="softmax", kernel_regularizer=kr.l2())(layer)

        model = k.Model(inputs=X, outputs=D_out)
        fidmodel = k.Model(inputs=X, outputs=layer)
    return model, fidmodel


def train_classifier():
    (x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()
    x_train = np.expand_dims((x_train / 128) - 1, axis=-1)
    x_test  = np.expand_dims((x_test / 128) - 1, axis=-1)
    y_train = k.utils.to_categorical(y_train)
    y_test = k.utils.to_categorical(y_test)

    model, fidmodel = create_classifier()
    model.compile("adam", "categorical_crossentropy", ["accuracy"])

    callbacks = [
        k.callbacks.TensorBoard("./runs/fid/"),
    ]

    model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test), callbacks=callbacks)
    model.evaluate(x_test, y_test)
    fidmodel.save("./fidmodel.hdf5")

if __name__ == "__main__":
    train_classifier()