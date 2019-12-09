import os
import random
import sys
import warnings

import progressbar

from metrics import create_fid_func

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers as kl

epochs = 1500
batch_size = 32
lr = 0.0002
noise_std = 0.01


def build_dy(Xt, Yt):

    with tf.name_scope("Dy"):
        X = kl.Input((28, 28, 1), tensor=Xt, name="X")
        Y = kl.Input((10,), tensor=Yt, name="Y")

        layer = kl.GaussianNoise(stddev=0.1)(X)

        # Discriminator
        layer = kl.Conv2D(
            filters=64,
            kernel_size=5,
            padding="same",
            strides=2)(layer)
        layer = kl.LeakyReLU()(layer)
        layer = kl.Dropout(0.3)(layer)
        layer = kl.BatchNormalization()(layer)

        layer = kl.Conv2D(
            filters=128,
            kernel_size=5,
            padding="same",
            strides=2)(layer)
        layer = kl.LeakyReLU()(layer)
        layer = kl.Dropout(0.3)(layer)
        layer = kl.BatchNormalization()(layer)

        layer = kl.Flatten()(layer)
        layer = kl.concatenate([layer, Y], axis=-1)
        layer = kl.Dense(980)(layer)
        layer = kl.LeakyReLU()(layer)
        layer = kl.BatchNormalization()(layer)
        D_out = kl.Dense(1, activation="sigmoid")(layer)

        return k.Model(inputs=[X, Y], outputs=D_out)


def build_g(Yt, Zt):
    with tf.name_scope("Gen"):

        Z = kl.Input((100,), tensor=Zt, name="Z")
        Y = kl.Input((10,), tensor=Yt, name="Y")

        layer = kl.concatenate([Z, Y])
        layer = kl.Dense(128*7*7, activation="relu")(layer)
        layer = kl.BatchNormalization()(layer)
        layer = kl.Reshape((7, 7, 128))(layer)

        layer = kl.UpSampling2D(size=(2, 2))(layer)
        layer = kl.Conv2D(
            filters=64,
            kernel_size=5,
            padding="same")(layer)
        layer = kl.LeakyReLU()(layer)
        layer = kl.BatchNormalization()(layer)

        layer = kl.UpSampling2D(size=(2, 2))(layer)

        G_out = kl.Conv2D(
            filters=1,
            kernel_size=5,
            activation="tanh",
            padding="same")(layer)

    return k.Model(inputs=[Z, Y], outputs=G_out)


def run(logs_dir, checkpoints_dir, checkpoints=False):
    with tf.Session() as sess:

        # Preparing the dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = np.expand_dims((x_train / 128) - 1, axis=-1)
        x_test = np.expand_dims((x_test / 128) - 1, axis=-1)

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        def get_noise():
            return np.random.uniform(-1, 1, (batch_size, 100))

        # Preparing the models for the FID computation
        fid = create_fid_func("fid_models/mnist.hdf5", sess)

        # Building models
        x = tf.placeholder(tf.float32, (None, 28, 28, 1), name="X")
        y = tf.placeholder(tf.float32, (None, 10), name="C")
        z = tf.placeholder(tf.float32, (None, 100,), name="Z")

        Dy = build_dy(x, y)
        G = build_g(y, z)

        # Objectives
        with tf.name_scope("L_xy"):
            L_xy = tf.reduce_mean(-tf.log(Dy([x, y]) + 1e-8)) + \
                   tf.reduce_mean(-tf.log(1 - Dy([G([y, z]), y]) + 1e-8))

        with tf.name_scope("L_xy_nonsat"):
            L_xy_nonsat = tf.reduce_mean(-tf.log(Dy([G([z, y]), y]) + 1e-8))

        # Optimizers
        adam = tf.train.AdamOptimizer
        Dy_cost = adam(lr).minimize(L_xy, var_list=Dy.trainable_weights)
        G_cost = adam(lr).minimize(L_xy_nonsat, var_list=G.trainable_weights)

        # Logging costs
        dycs = tf.summary.scalar("Dy_cost", L_xy)
        gcs = tf.summary.scalar("G_cost", L_xy_nonsat)

        # Logging images
        imgpl = tf.placeholder(tf.float32, shape=(1, 28, 28, 1))
        imgsummary = tf.summary.image("Generated", imgpl)

        # Setting up the training
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(
            logs_dir + os.sep + "supahgan", tf.get_default_graph())
        os.mkdir(checkpoints_dir)
        writer.flush()

        # Do the actual training
        for epoch in range(epochs):

            # Shuffling the data
            rng_state = np.random.get_state()
            np.random.shuffle(x_train)
            np.random.set_state(rng_state)
            np.random.shuffle(y_train)

            print("Epoch " + str(epoch))
            for it in progressbar.ProgressBar(maxvalue=len(x_train), redirect_stdout=True)(
                    range(0, len(x_train), batch_size * 6)):

                t = len(x_train) * epoch + it

                # Training Dy
                xbatch = x_train[it:it + batch_size]
                ybatch = y_train[it:it + batch_size]
                noise = get_noise()
                if len(xbatch) == batch_size:
                    _, dyout = sess.run([Dy_cost, dycs], feed_dict={x: xbatch, y: ybatch, z: noise})
                    writer.add_summary(dyout, t)

                # Training G
                xbatch = x_train[it + batch_size * 5:it + batch_size * 6]
                ybatch = y_train[it + batch_size * 5:it + batch_size * 6]
                noise = get_noise()
                if len(xbatch) == batch_size:
                    _, gout = sess.run([G_cost, gcs], feed_dict={x: xbatch, y: ybatch, z: noise})
                    writer.add_summary(gout, t)

            # Epoch ended, logging metrics on validation set
            gener_list = np.empty((0, 28, 28, 1))
            print("Evaluating on validation")
            bar = progressbar.ProgressBar(maxvalue=100, redirect_stdout=True)
            for i in bar(range(0, len(x_test), batch_size)):
                xbatch = x_test[i:i + batch_size]
                ybatch = y_test[i:i + batch_size]

                if len(xbatch) == batch_size:
                    noise = get_noise()
                    generated = sess.run(G.output, feed_dict={x: xbatch, y: ybatch, z: noise})

                    gener_list = np.concatenate([gener_list, generated])

            metricsout = tf.Summary(value=[
                tf.Summary.Value(tag="FID", simple_value=fid(x_test, gener_list)),
            ])

            # Logging images
            print("Logging images")
            imgout = sess.run(imgsummary, feed_dict={imgpl: np.expand_dims(gener_list[random.randint(0, len(gener_list))], axis=0)})

            # Writing all logs in tensorboard
            writer.add_summary(metricsout, epoch)
            writer.add_summary(imgout, epoch)
            writer.flush()

            # Saving weights (if needed)
            if checkpoints:
                G.save(checkpoints_dir + os.sep + "G_" + str(epoch) + ".hdf5")
                Dy.save(checkpoints_dir + os.sep + "Dy_" + str(epoch) + ".hdf5")

        # Run end
        writer.close()
    tf.reset_default_graph()


def main():
    name = "notsupahgan" + sys.argv[1]
    print("\nRunning " + name + "\n")

    os.mkdir("./runs/" + name)
    checkpoints_dir = "./runs/" + name + "/checkpoints/"
    logs_dir = "./runs/" + name + "/logs/"

    run(logs_dir, checkpoints_dir, checkpoints=False)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
