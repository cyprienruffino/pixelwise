import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import regularizers as kr


def create_classifier():

    with tf.name_scope("Disc"):
        X = kl.Input((32, 32, 3), name="X")
        layer = X

        for l in range(4):
            layer = kl.Conv2D(
                filters=32 * (2**l),
                kernel_size=3,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_regularizer=kr.l2())(layer)
            layer = kl.BatchNormalization()(layer)
            layer = kl.Conv2D(
                filters=32 * (2**l),
                kernel_size=3,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_regularizer=kr.l2())(layer)
            layer = kl.BatchNormalization()(layer)
            layer = kl.MaxPool2D()(layer)
            layer = kl.Dropout(0.5)(layer)

        layer = kl.Flatten()(layer)
        layer = kl.Dense(600, kernel_regularizer=kr.l2())(layer)
        layer = kl.LeakyReLU()(layer)
        layer = kl.Dropout(0.5)(layer)
        D_out = kl.Dense(10, activation="softmax",
                         kernel_regularizer=kr.l2())(layer)

        model = k.Model(inputs=X, outputs=D_out)
        fidmodel = k.Model(inputs=X, outputs=layer)
    return model, fidmodel


def train_classifier():
    (x_train, y_train), (x_test, y_test) = k.datasets.cifar10.load_data()
    x_train = (x_train / 128) - 1
    x_test = (x_test / 128) - 1
    y_train = k.utils.to_categorical(y_train)
    y_test = k.utils.to_categorical(y_test)

    model, fidmodel = create_classifier()
    model.compile(k.optimizers.Adam(0.0005), "categorical_crossentropy", ["accuracy"])

    callbacks = [
        k.callbacks.TensorBoard("./runs/fid/")
    ]

    model.fit(x_train, y_train, batch_size=32, epochs=30,
              validation_data=(x_test, y_test), callbacks=callbacks)
    model.evaluate(x_test, y_test)
    fidmodel.save("./fidmodel.hdf5")


if __name__ == "__main__":
    train_classifier()
