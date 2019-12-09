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

epochs = 400
batch_size = 32
lr = 0.000005
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


def build_dz(Xt, Zt):
    with tf.name_scope("Dz"):
        X = kl.Input((28, 28, 1), tensor=Xt, name="X")
        Z = kl.Input((100,), tensor=Zt, name="Z")

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
        layer = kl.concatenate([layer, Z], axis=-1)
        layer = kl.Dense(980)(layer)
        layer = kl.LeakyReLU()(layer)
        layer = kl.BatchNormalization()(layer)
        D_out = kl.Dense(1, activation="sigmoid")(layer)

        return k.Model(inputs=[X, Z], outputs=D_out)


def build_g(Et):
    with tf.name_scope("Gen"):
        E = kl.Input((100,), tensor=Et, name="Z")

        layer = kl.Dense(128 * 7 * 7, activation="relu")(E)
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

    return k.Model(inputs=E, outputs=G_out)


def build_c(Xt):
    with tf.name_scope("C"):
        X = kl.Input((28, 28, 1), tensor=Xt, name="X")
        layer = X

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
        layer = kl.Dense(980)(layer)
        layer = kl.LeakyReLU()(layer)
        layer = kl.BatchNormalization()(layer)
        D_out = kl.Dense(10, activation="softmax")(layer)

        model = k.Model(inputs=X, outputs=D_out)
    return model


def build_i(Xt):
    with tf.name_scope("I"):
        X = kl.Input((28, 28, 1), tensor=Xt, name="X")
        layer = X

        for l in range(3):
            layer = kl.Conv2D(
                filters=64 * (2 ** l),
                kernel_size=3,
                padding="same",
                activation="relu")(layer)
            layer = kl.Conv2D(
                filters=64 * (2 ** l),
                kernel_size=3,
                padding="same",
                activation="relu")(layer)
            layer = kl.MaxPool2D()(layer)
            layer = kl.BatchNormalization()(layer)

        layer = kl.Flatten()(layer)
        layer = kl.Dense(980)(layer)
        layer = kl.LeakyReLU()(layer)
        layer = kl.BatchNormalization()(layer)
        D_out = kl.Dense(100, activation="tanh")(layer)

        model = k.Model(inputs=X, outputs=D_out)
    return model


def build_e(Yt, Zt):
    with tf.name_scope("E"):
        Z = kl.Input((100,), tensor=Zt, name="Z")
        Y = kl.Input((10,), tensor=Yt, name="Y")
        layer = kl.concatenate([Z, Y])

        layer = kl.Dense(256, activation="relu")(layer)
        layer = kl.Dense(256, activation="relu")(layer)
        D_out = kl.Dense(100, activation="tanh")(layer)

        model = k.Model(inputs=[Y, Z], outputs=D_out)
    return model


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

        E = build_e(y, z)
        Dy = build_dy(x, y)
        Dz = build_dz(x, z)
        C = build_c(x)
        I = build_i(x)
        G = build_g(E.output)

        # Objectives
        with tf.name_scope("L_xy"):
            L_xy = tf.reduce_mean(-tf.log(Dy([x, C(x)]) + 1e-8)) + \
                   tf.reduce_mean(-tf.log(1 - Dy([G(E([y, z])), y]) + 1e-8))

        with tf.name_scope("L_xz"):
            L_xz = tf.reduce_mean(-tf.log(Dz([x, I(x)]) + 1e-8)) + \
                   tf.reduce_mean(-tf.log(1 - Dz([G(E([y, z])), z]) + 1e-8))

        with tf.name_scope("L_infer"):
            L_infer = tf.reduce_mean(tf.square(z - I(G(E([z, y])))))

        with tf.name_scope("L_class_rec"):
            L_class_rec = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y, tf.log(C(x) + 1e-8)), 1))

        with tf.name_scope("L_class_true"):
            L_class_true = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y, tf.log(C(G(E([z, y]))) + 1e-8)), 1))

        with tf.name_scope("L_xy_nonsat"):
            L_xy_nonsat = tf.reduce_mean(-tf.log(Dy([G(E([z, y])), y]) + 1e-8))

        with tf.name_scope("L_xz_nonsat"):
            L_xz_nonsat = tf.reduce_mean(-tf.log(Dz([G(E([z, y])), z]) + 1e-8))

        # Optimizers
        adam = tf.train.AdamOptimizer
        Dy_cost = adam(lr).minimize(L_xy, var_list=Dy.trainable_weights)
        Dz_cost = adam(lr).minimize(L_xz, var_list=Dz.trainable_weights)
        C_cost = adam(lr).minimize(L_xy + L_class_true + L_class_rec, var_list=C.trainable_weights)
        I_cost = adam(lr).minimize(L_xz + L_infer, var_list=I.trainable_weights)
        E_cost = adam(lr).minimize(L_xy_nonsat + L_xz_nonsat + L_class_true + L_class_rec + L_infer,
                                   var_list=E.trainable_weights)
        G_cost = adam(lr).minimize(L_xy_nonsat + L_xz_nonsat + L_class_true + L_class_rec + L_infer,
                                   var_list=G.trainable_weights)

        # Logging costs
        dzcs = tf.summary.scalar("Dz_cost", L_xz)
        dycs = tf.summary.scalar("Dy_cost", L_xy)
        ccs = tf.summary.scalar("C_cost", L_xy + L_class_true + L_class_rec)
        ics = tf.summary.scalar("I_cost", L_xz + L_infer)
        ecs = tf.summary.scalar("E_cost", L_xy_nonsat + L_xz_nonsat + L_class_true + L_infer)
        gcs = tf.summary.scalar("G_cost", L_xy_nonsat + L_xz_nonsat + L_class_true + L_infer)

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

                # Training Dz
                xbatch = x_train[it + batch_size:it + batch_size * 2]
                ybatch = y_train[it + batch_size:it + batch_size * 2]
                noise = get_noise()
                if len(xbatch) == batch_size:
                    _, dzout = sess.run([Dz_cost, dzcs], feed_dict={x: xbatch, y: ybatch, z: noise})
                    writer.add_summary(dzout, t)

                # Training C
                xbatch = x_train[it + batch_size * 2:it + batch_size * 3]
                ybatch = y_train[it + batch_size * 2:it + batch_size * 3]
                noise = get_noise()
                if len(xbatch) == batch_size:
                    _, cout = sess.run([C_cost, ccs], feed_dict={x: xbatch, y: ybatch, z: noise})
                    writer.add_summary(cout, t)

                # Training I
                xbatch = x_train[it + batch_size * 3:it + batch_size * 4]
                ybatch = y_train[it + batch_size * 3:it + batch_size * 4]
                noise = get_noise()
                if len(xbatch) == batch_size:
                    _, iout = sess.run([I_cost, ics], feed_dict={x: xbatch, y: ybatch, z: noise})
                    writer.add_summary(iout, t)

                # Training E
                xbatch = x_train[it + batch_size * 4:it + batch_size * 5]
                ybatch = y_train[it + batch_size * 4:it + batch_size * 5]
                noise = get_noise()
                if len(xbatch) == batch_size:
                    _, eout = sess.run([E_cost, ecs], feed_dict={x: xbatch, y: ybatch, z: noise})
                    writer.add_summary(eout, t)

                # Training G
                xbatch = x_train[it + batch_size * 5:it + batch_size * 6]
                ybatch = y_train[it + batch_size * 5:it + batch_size * 6]
                noise = get_noise()
                if len(xbatch) == batch_size:
                    _, gout = sess.run([G_cost, gcs], feed_dict={x: xbatch, y: ybatch, z: noise})
                    writer.add_summary(gout, t)

            # Epoch ended, logging metrics on validation set
            gener_list = np.empty((0, 28, 28, 1))
            classif_list_true = np.empty((0, 10))
            classif_list_gen = np.empty((0, 10))
            infer_list_gen = np.empty((0, 100))
            noises = np.empty((0, 100))
            real_y = np.empty((0, 10))

            print("Evaluating on validation")
            bar = progressbar.ProgressBar(maxvalue=100, redirect_stdout=True)
            for i in bar(range(0, len(x_test), batch_size)):
                xbatch = x_test[i:i + batch_size]
                ybatch = y_test[i:i + batch_size]

                if len(xbatch) == batch_size:
                    noise = get_noise()
                    generated = sess.run(G.output, feed_dict={x: xbatch, y: ybatch, z: noise})
                    classified_true = sess.run(C.output, feed_dict={x: xbatch})
                    classified_gen = sess.run(C.output, feed_dict={x: generated})
                    infered_gen = sess.run(I.output, feed_dict={x: generated})

                    classif_list_true = np.concatenate([classif_list_true, classified_true])
                    infer_list_gen = np.concatenate([infer_list_gen, infered_gen])
                    classif_list_gen = np.concatenate([classif_list_gen, classified_gen])
                    gener_list = np.concatenate([gener_list, generated])
                    noises = np.concatenate([noises, noise])
                    real_y = np.concatenate([real_y, ybatch])

            metricsout = tf.Summary(value=[
                tf.Summary.Value(tag="FID", simple_value=fid(x_test, gener_list)),
                tf.Summary.Value(tag="MSE_I_gen", simple_value=np.mean((noises - infer_list_gen) ** 2)),
                tf.Summary.Value(tag="MSE_C_gen", simple_value=np.mean((real_y - classif_list_gen) ** 2)),
                tf.Summary.Value(tag="MSE_C_true", simple_value=np.mean((real_y - classif_list_true) ** 2))
            ])

            # Logging images
            print("Logging images")
            imgout = sess.run(imgsummary,
                              feed_dict={imgpl: np.expand_dims(gener_list[random.randint(0, len(gener_list))], axis=0)})

            # Writing all logs in tensorboard
            writer.add_summary(metricsout, epoch)
            writer.add_summary(imgout, epoch)
            writer.flush()

            # Saving weights (if needed)
            if checkpoints:
                G.save(checkpoints_dir + os.sep + "G_" + str(epoch) + ".hdf5")
                Dy.save(checkpoints_dir + os.sep + "Dy_" + str(epoch) + ".hdf5")
                Dz.save(checkpoints_dir + os.sep + "Dz_" + str(epoch) + ".hdf5")
                C.save(checkpoints_dir + os.sep + "C_" + str(epoch) + ".hdf5")
                I.save(checkpoints_dir + os.sep + "I_" + str(epoch) + ".hdf5")
                E.save(checkpoints_dir + os.sep + "E_" + str(epoch) + ".hdf5")

        # Run end
        writer.close()
    tf.reset_default_graph()


def main():
    name = "supahgan" + sys.argv[1]
    print("\nRunning " + name + "\n")

    os.mkdir("./runs/" + name)
    checkpoints_dir = "./runs/" + name + "/checkpoints/"
    logs_dir = "./runs/" + name + "/logs/"

    run(logs_dir, checkpoints_dir, checkpoints=False)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
