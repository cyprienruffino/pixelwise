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


def run(cfg, dataset_path, logs_dir, checkpoints_dir, checkpoints=True):
    with tf.Session() as sess:
        get_data = from_files.get_data_provider(dataset_path, cfg.batch_size)
        get_gen = from_files.get_genconst_provider(dataset_path, cfg.batch_size)

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
        Z = tf.placeholder(tf.float32, (None, cfg.zx, cfg.zx, cfg.nz,), name="Z")
        Z2 = tf.placeholder(tf.float32, (None, cfg.zx, cfg.zx, cfg.nz,), name="Z2")
        X = tf.placeholder(tf.float32, (None, cfg.npx, cfg.npx, cfg.channels,), name="X")
        X2 = tf.placeholder(tf.float32, (None, cfg.npx, cfg.npx, cfg.channels,), name="X2")
        Cf = tf.placeholder(tf.float32, (None, cfg.npx, cfg.npx, cfg.channels,), name="Cf")

        D = cfg.discriminator(X, X2, **cfg.disc_args)
        G = cfg.generator(Z, Cf, **cfg.gen_args)

        G_out = G([Z, Cf])
        G2_out = G([Z2, Cf])

        with tf.name_scope("DG"):
            DG = D([G_out, G2_out])

        # Objectives
        with tf.name_scope("D_real_objective"):
            D_real_objective = tf.reduce_mean(-tf.log(D.output + 1e-8))

        with tf.name_scope("D_fake_objective"):
            D_fake_objective = tf.reduce_mean(-tf.log(1 - DG + 1e-8))

        with tf.name_scope("D_objective"):
            D_objective = D_real_objective + D_fake_objective

        with tf.name_scope("MSE_cost"):
            mask = tf.cast(tf.greater(tf.abs(Cf), 0), tf.float32)  # Getting binary mask of Cf
            nb_const = tf.reduce_sum(mask)
            masked_image = mask * G_out
            mse = tf.reduce_sum(tf.square(Cf - masked_image)) / nb_const
            masked_image = mask * G2_out
            mse += tf.reduce_sum(tf.square(Cf - masked_image)) / nb_const

        with tf.name_scope("G_objective"):
            G_real_objective = tf.reduce_mean(-tf.log(DG + 1e-8))
            G_objective = G_real_objective + (cfg.lmbda * mse)

        # Optimizers
        D_optimizer = cfg.disc_optimizer(**cfg.disc_optimizer_args)
        G_optimizer = cfg.gen_optimizer(**cfg.gen_optimizer_args)

        D_cost = D_optimizer.minimize(D_objective, var_list=D.trainable_weights)
        G_cost = G_optimizer.minimize(G_objective, var_list=G.trainable_weights)

        # Logging costs
        msesumm = tf.summary.scalar("MSE_train", mse)
        drealcostsumm = tf.summary.scalar("D_real_cost", D_real_objective)
        dfakecostsumm = tf.summary.scalar("D_fake_cost", D_fake_objective)
        gcostsumm = tf.summary.scalar("G_cost", G_real_objective)

        # Logging images
        consttrueimgpl = tf.placeholder(tf.float32, shape=(1, cfg.npx, cfg.npx, 3))
        trueimgpl = tf.placeholder(tf.float32, shape=(1, cfg.npx, cfg.npx, cfg.channels))
        imgpl = tf.placeholder(tf.float32, shape=(1, cfg.npx, cfg.npx, cfg.channels))
        constimgpl = tf.placeholder(tf.float32, shape=(1, cfg.npx, cfg.npx, 3))
        imgpl2 = tf.placeholder(tf.float32, shape=(1, cfg.npx, cfg.npx, cfg.channels))
        constimgpl2 = tf.placeholder(tf.float32, shape=(1, cfg.npx, cfg.npx, 3))

        imgsummaries = [
            tf.summary.image("Generated_const", constimgpl),
            tf.summary.image("Ground_truth_const", consttrueimgpl),
            tf.summary.image("Generated", imgpl),
            tf.summary.image("Ground_truth", trueimgpl),
            tf.summary.image("Generated_const_2", constimgpl2),
            tf.summary.image("Generated_2", imgpl2)
        ]

        imgsumm = tf.summary.merge(imgsummaries)

        # Setting up the training
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(
            logs_dir + os.sep + cfg.name, tf.get_default_graph())
        os.mkdir(checkpoints_dir)
        writer.flush()

        # Do the actual training
        for epoch in range(cfg.epochs):
            bar = progressbar.ProgressBar(maxvalue=cfg.epoch_iters, redirect_stdout=True)
            print("----------------------------------------------------Epoch " + str(
                epoch) + "----------------------------------------------------")

            for it in bar(range(int(cfg.dataset_size / cfg.batch_size))):
                # Training D
                x_real, c_real = get_data()
                x_real2, _ = get_data()
                c_fake = get_gen()
                noise = get_noise(cfg.batch_size)
                noise2 = get_noise(cfg.batch_size)
                _, drealout, dfakeout = sess.run([D_cost, drealcostsumm, dfakecostsumm],
                                                 feed_dict={X: x_real, X2:  x_real2, Z: noise, Z2: noise2, Cf: c_fake})

                # Training G
                c_fake = get_gen()
                noise = get_noise(cfg.batch_size)
                noise2 = get_noise(cfg.batch_size)
                _, mseout, gout = sess.run([G_cost, msesumm, gcostsumm], feed_dict={Z: noise, Z2: noise2, Cf: c_fake})

                # Logging losses
                t = int(cfg.dataset_size / cfg.batch_size) * epoch + it
                writer.add_summary(drealout, t)
                writer.add_summary(dfakeout, t)
                writer.add_summary(gout, t)
                writer.add_summary(mseout, t)

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
                    generated.extend(list(sess.run(G_out, feed_dict={Z: noise, Cf: consts})))
                    valid_data.extend(list(real_imgs))
                    valid_consts.extend(list(consts))

            generated = np.asarray(generated)
            valid_data = np.asarray(valid_data)
            valid_consts = np.asarray(valid_consts)

            for m in cfg.metrics:
                if m not in curmets:
                    curmets[m] = []
                curmets[m].append(cfg.metrics[m](valid_data, generated))

            metricslist = [
                tf.Summary.Value(tag="MSE", simple_value=metrics.mse(generated, valid_consts)),
                tf.Summary.Value(tag="FID", simple_value=fid(valid_data, generated))
            ]

            for m in curmets.keys():
                metricslist.append(tf.Summary.Value(tag=m, simple_value=np.mean(curmets[m])))
            metricsout = tf.Summary(value=metricslist)

            # Logging images
            print("Logging images")
            true_img = np.expand_dims(x_real[0], axis=0)
            const = np.expand_dims(c_real[0], axis=0)
            noise = get_noise(1)
            img = sess.run(G_out, feed_dict={Z: noise, Cf: const})

            noise = get_noise(1)
            img2 = sess.run(G_out, feed_dict={Z: noise, Cf: const})

            imgout = sess.run(
                imgsumm, feed_dict={
                    trueimgpl: true_img,
                    imgpl: img,
                    constimgpl: log.constraints_image(img, const),
                    imgpl2: img,
                    constimgpl2: log.constraints_image(img2, const),
                    consttrueimgpl: log.constraints_image(true_img, const)
                })
            writer.flush()

            # Writing all logs as tensorboard
            writer.add_summary(metricsout, epoch)
            writer.add_summary(imgout, epoch)
            writer.flush()

            # Saving weights
            if checkpoints:
                G.save(checkpoints_dir + os.sep + "G_" + str(epoch) + ".hdf5", include_optimizer=False)
                D.save(checkpoints_dir + os.sep + "D_" + str(epoch) + ".hdf5", include_optimizer=False)

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

main