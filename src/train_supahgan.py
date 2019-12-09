import os
import shutil
import sys
import warnings

import progressbar

import metrics
from datasets import from_files
# from datasets import from_files
from metrics import create_fid_func
from utils import log
from utils.config import loadconfig

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import tensorflow as tf


def run(cfg, dataset_path, logs_dir, checkpoints_dir, checkpoints=False):
    with tf.Session() as sess:

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = np.concatenate([x_train, x_test])
        x_train = (x_train / 128) - 1

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        if cfg.validation:
            get_valid_data = from_files.get_valid_provider(dataset_path, cfg.batch_size)
            get_valid_const = from_files.get_validconst_provider(dataset_path, cfg.batch_size)
        else:
            get_valid_data = None
            get_valid_const = None

        get_noise = cfg.noise_provider(**cfg.noise_provider_args)

        if cfg.fid_model is not None:
            fid = create_fid_func(cfg.fid_model, sess)
        else:
            fid = lambda x, y: 0

        # Building models
        z = tf.placeholder(tf.float32, (None, 100,), name="Z")
        x = tf.placeholder(tf.float32, (None, 28, 28, 1), name="X")
        y = tf.placeholder(tf.float32, (None, 10), name="C")

        E = cfg.encoder(z, y, **cfg.enc_args)
        Dy = cfg.disc_y(x, y, **cfg.disc_y_args)
        Dz = cfg.disc_z(x, z, **cfg.disc_z_args)
        C = cfg.classifier(x, **cfg.class_args)
        I = cfg.inferer(x, **cfg.inf_args)
        G = cfg.generator(E.output, **cfg.gen_args)

        # Objectives
        with tf.name_scope("L_xy"):
            L_xy = tf.reduce_mean(-tf.log(Dy(x, C(x)) + 1e-8)) - tf.reduce_mean(-tf.log(1 - Dy(G(E(y, z)), y) + 1e-8))

        with tf.name_scope("L_xz"):
            L_xz = tf.reduce_mean(-tf.log(Dz(x, I(x)) + 1e-8)) - tf.reduce_mean(-tf.log(1 - Dz(G(E(y, z)), z) + 1e-8))

        with tf.name_scope("L_infer"):
            L_infer = tf.reduce_mean(tf.square(z - I(G(E(z, y)))))

        with tf.name_scope("L_classif"):
            L_classif = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y, tf.log(C(G(E(z, y))))), 1))

        with tf.name_scope("L_xy_nonsat"):
            L_xy_nonsat = tf.reduce_mean(-tf.log(Dy(G(E(z, y)), y) + 1e-8))

        with tf.name_scope("L_xz_nonsat"):
            L_xz_nonsat = tf.reduce_mean(-tf.log(Dz(G(E(z, y)), z) + 1e-8))

        # Optimizers
        Dz_optimizer = cfg.disc_optimizer(**cfg.disc_z_optimizer_args)
        Dy_optimizer = cfg.gen_optimizer(**cfg.disc_y_optimizer_args)
        G_optimizer = cfg.gen_optimizer(**cfg.gen_optimizer_args)
        E_optimizer = cfg.enc_optimizer(**cfg.enc_optimizer_args)
        C_optimizer = cfg.clas_optimizer(**cfg.clas_optimizer_args)
        I_optimizer = cfg.inf_optimizer(**cfg.inf_optimizer_args)

        Dz_cost = Dz_optimizer.minimize(L_xz, var_list=Dz.trainable_weights)
        Dy_cost = Dy_optimizer.minimize(L_xy, var_list=Dy.trainable_weights)
        C_cost = C_optimizer.minimize(L_xy + L_classif, var_list=C.trainable_weights)
        I_cost = I_optimizer.minimize(L_xz + L_infer, var_list=I.trainable_weights)
        E_cost = E_optimizer.minimize(L_xy + L_xz + L_classif + L_infer, var_list=E.trainable_weights)
        G_cost = G_optimizer.minimize(L_xy + L_xz + L_classif + L_infer, var_list=G.trainable_weights)

        # Logging costs
        dzcs = tf.summary.scalar("Dz_cost", L_xz)
        dycs = tf.summary.scalar("Dy_cost", L_xy)
        ccs = tf.summary.scalar("C_cost", L_xy + L_classif)
        ics = tf.summary.scalar("I_cost", L_xz + L_infer)
        ecs = tf.summary.scalar("E_cost", L_xy_nonsat + L_xz_nonsat + L_classif + L_infer)
        gcs = tf.summary.scalar("G_cost", L_xy_nonsat + L_xz_nonsat + L_classif + L_infer)

        # Logging images
        imgpl = tf.placeholder(tf.float32, shape=(28, 28, 1))
        imgsummary = tf.summary.image("Generated", G.output)

        # Setting up the training
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(
            logs_dir + os.sep + cfg.name, tf.get_default_graph())
        os.mkdir(checkpoints_dir)
        writer.flush()

        # Do the actual training
        for epoch in range(cfg.epochs):
            bar = progressbar.ProgressBar(maxvalue=cfg.epoch_iters, redirect_stdout=True)
            print("Epoch " + str(epoch))

            for it in bar(range(0, len(x_train), cfg.batch_size*6)):
                # Training Dy
                xbatch = x_train[it:it + cfg.batch_size]
                ybatch = y_train[it:it + cfg.batch_size]
                noise = get_noise(cfg.batch_size)
                _, dyout = sess.run([Dy_cost, dycs], feed_dict={x: xbatch, z: noise, y: ybatch})

                # Training Dz
                xbatch = x_train[it:it + cfg.batch_size*2]
                ybatch = y_train[it:it + cfg.batch_size*2]
                noise = get_noise(cfg.batch_size)
                _, dzout = sess.run([Dz_cost, dzcs], feed_dict={x: xbatch, z: noise, y: ybatch})

                # Training C
                xbatch = x_train[it:it + cfg.batch_size*3]
                ybatch = y_train[it:it + cfg.batch_size*3]
                noise = get_noise(cfg.batch_size)
                _, cout = sess.run([C_cost, ccs], feed_dict={x: xbatch, z: noise, y: ybatch})

                # Training I
                xbatch = x_train[it:it + cfg.batch_size*4]
                ybatch = y_train[it:it + cfg.batch_size*4]
                noise = get_noise(cfg.batch_size)
                _, iout = sess.run([I_cost, ics], feed_dict={x: xbatch, z: noise, y: ybatch})

                # Training E
                xbatch = x_train[it:it + cfg.batch_size*5]
                ybatch = y_train[it:it + cfg.batch_size*5]
                noise = get_noise(cfg.batch_size)
                _, eout = sess.run([E_cost, ecs], feed_dict={x: xbatch, z: noise, y: ybatch})

                # Training G
                xbatch = x_train[it:it + cfg.batch_size*6]
                ybatch = y_train[it:it + cfg.batch_size*6]
                noise = get_noise(cfg.batch_size)
                _, gout = sess.run([G_cost, gcs], feed_dict={x: xbatch, z: noise, y: ybatch})

                # Logging losses
                t = len(x_train) * epoch + it
                writer.add_summary(dyout, t)
                writer.add_summary(dzout, t)
                writer.add_summary(cout, t)
                writer.add_summary(iout, t)
                writer.add_summary(eout, t)
                writer.add_summary(gout, t)

            # Epoch ended
            # Logging metrics on validation set
            curmets = {}
            generated = []
            valid_data = []
            valid_consts = []
            bar = progressbar.ProgressBar(maxvalue=int(cfg.valid_size / cfg.batch_size), redirect_stdout=True)
            print("Generating on validation")
            for i in bar(range(int(cfg.valid_size / cfg.batch_size))):
                real_imgs = get_valid_data()
                consts = get_valid_const()
                if len(real_imgs) == cfg.batch_size:
                    noise = get_noise(cfg.batch_size)
                    generated.extend(list(sess.run(G.output, feed_dict={z: noise, x: real_imgs, y: consts})))
                    valid_data.extend(list(real_imgs))
                    valid_consts.extend(list(consts))

            generated = np.asarray(generated)
            valid_data = np.asarray(valid_data)

            for m in cfg.metrics:
                if m not in curmets:
                    curmets[m] = []
                curmets[m].append(cfg.metrics[m](valid_data, generated))

            metricslist = [tf.Summary.Value(tag="FID", simple_value=fid(valid_data, generated))]

            for m in curmets.keys():
                metricslist.append(tf.Summary.Value(tag=m, simple_value=np.mean(curmets[m])))
            metricsout = tf.Summary(value=metricslist)

            # Logging images
            print("Logging images")

            imgout = sess.run(imgsummary, feed_dict={imgpl: generated[0]})

            # Writing all logs as tensorboard
            writer.add_summary(metricsout, epoch)
            writer.add_summary(imgout, epoch)
            writer.flush()

            # Saving weights
            if checkpoints:
                G.save(checkpoints_dir + os.sep + "G_" + str(epoch) + ".hdf5", include_optimizer=False)
                Dy.save(checkpoints_dir + os.sep + "Dy_" + str(epoch) + ".hdf5", include_optimizer=False)
                Dz.save(checkpoints_dir + os.sep + "Dz_" + str(epoch) + ".hdf5", include_optimizer=False)
                C.save(checkpoints_dir + os.sep + "C_" + str(epoch) + ".hdf5", include_optimizer=False)
                I.save(checkpoints_dir + os.sep + "I_" + str(epoch) + ".hdf5", include_optimizer=False)
                E.save(checkpoints_dir + os.sep + "E_" + str(epoch) + ".hdf5", include_optimizer=False)

        # Run end
        writer.close()
    tf.reset_default_graph()


def do_run(filepath, dataset_path, extension=""):
    name = filepath.split('/')[-1].replace('.py', '') + extension
    config = loadconfig(filepath)

    print("\nRunning " + name + "\n")

    os.mkdir("./runs/" + name)
    shutil.copy2(filepath, './runs/' + name + "/rcgan_config.py")
    checkpoints_dir = "./runs/" + name + "/checkpoints/"
    logs_dir = "./runs/" + name + "/logs/"

    run(config, dataset_path, logs_dir, checkpoints_dir, checkpoints=True)


def main():
    if len(sys.argv) < 3:
        print("Usage : python train_rcgan.py path_to_config_file path_to_image_folder [run_name]")
        exit(1)

    if len(sys.argv) == 4:
        extension = sys.argv[3]
    else:
        extension = ""

    filepath = sys.argv[1]
    dataset_path = sys.argv[2]
    print(filepath)

    if os.path.isfile(filepath):
        do_run(filepath, dataset_path, extension)
    else:
        for run in os.listdir(filepath):
            if ".py" in run:
                do_run(filepath + "/" + run, dataset_path)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
