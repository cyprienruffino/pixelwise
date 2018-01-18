import pickle
import time
import h5py
import progressbar
import os
import shutil

import numpy as np
from tensorflow import set_random_seed
from io import TextIOWrapper

from hvdconfig import Config


def train(sgancfg,
          data_provider,
          run_name=None,
          checkpoints_dir="./",
          logs_dir="./",
          samples_dir="./",
          progress_bar=True,
          use_tensorboard=True,
          plot_models=True,
          save_json=True,
          save_config_file=True,
          generate_png=True,
          generate_hdf5=True,
          D_path=None,
          G_path=None,
          DG_path=None,
          Adv_path=None,
          initial_epoch=0):

    if run_name is None:
        run_name = str(time.time())

    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    if not os.path.exists(samples_dir):
        os.mkdir(samples_dir)

    if generate_png:
        from PIL import Image

    # Loading the config file
    if type(sgancfg) == str or type(sgancfg) == TextIOWrapper:
        with open(sgancfg, "rb") as f:
            config = pickle.load(f)
    elif type(sgancfg) == Config:
        config = sgancfg
    else:
        raise TypeError(
            "sgancfg : unknown type. Must pass a path as a string, an opened file or a Config object"
        )

    # Seeding the random numbers generators
    np.random.seed(config.seed)
    set_random_seed(config.seed)

    print("Seeding done, importing Keras")

    # Importing Keras must be done after the seeding
    from kgan.gan import gan
    import kgan.constraints

    import horovod.keras as hvd
    import tensorflow as tf

    import keras.backend as K
    from keras.models import load_model
    from keras.utils import plot_model
    from keras.backend import get_session

    hvd.init()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=tfconfig))

    print(hvd.rank())

    if hvd.rank() == 0:
        main_thread = True
    else:
        main_thread = False

    # Load or create the model
    if D_path is not None and G_path is not None and DG_path is not None:
        custom_objects = {
            "wasserstein_true": config.loss_true,
            "wasserstein_fake": config.loss_fake,
            "wasserstein_gen": config.loss_gen,
            "Clip": kgan.constraints.Clip
        }

        G = load_model(G_path, custom_objects=custom_objects)
        D = load_model(D_path, custom_objects=custom_objects)
        DG = load_model(DG_path, custom_objects=custom_objects)
        Adv = load_model(Adv_path, custom_objects=custom_objects)
    else:
        optimizer = hvd.DistributedOptimizer(config.optimizer(config.optimizer_params))
        G = config.generator(config.convdims, config.nc, config.clip_weights,
                             config.c)
        D = config.discriminator(config.convdims, config.nc,
                                 config.clip_weights, config.c)
        D, G, DG, Adv = gan(D, G, config.loss_true, config.loss_fake,
                            config.loss_gen, optimizer)

    # Setting up the TensorBoard logger
    if use_tensorboard and main_thread:
        import tensorflow as tf
        writer = tf.summary.FileWriter(logs_dir + "/" + run_name + ".tblog",
                                       get_session().graph)

    # Plotting the models
    if plot_models and main_thread:
        plot_model(D, logs_dir + "/D.png")
        plot_model(G, logs_dir + "/G.png")
        plot_model(DG, logs_dir + "/DG.png")
        plot_model(Adv, logs_dir + "/Adv.png")

    # Saving models structure
    if save_json and main_thread:
        with open(logs_dir + "/D.json", "w") as f:
            f.write(D.to_json())
        with open(logs_dir + "/G.json", "w") as f:
            f.write(G.to_json())
        with open(logs_dir + "/DG.json", "w") as f:
            f.write(DG.to_json())
        with open(logs_dir + "/Adv.json", "w") as f:
            f.write(Adv.to_json())

    if save_config_file and main_thread:
        shutil.copy("hvdconfig.py", logs_dir + "/hvdconfig.py")

    # Sampling
    z_sample = np.random.uniform(-1., 1., (1, config.nz) +
                                 ((config.zx_sample, ) * config.convdims))

    # Do the actual training
    nb_iters = int(config.epoch_iters / hvd.size())
    hvd.broadcast_global_variables(0)
    for epoch in range(config.epochs):

        if main_thread:
            print("Epoch", epoch)
            if progress_bar:
                bar = progressbar.ProgressBar()
            else:
                bar = lambda x: x
        else:
            bar = lambda x: x

        G_losses = []
        D_losses = []
        D_fake_losses = []
        D_real_losses = []

        for it in bar(range(int(nb_iters / config.batch_size))):
            samples = next(data_provider)

            # Creating the input noise
            Znp = np.random.uniform(-1., 1., (config.batch_size, config.nz) +
                                    ((config.zx, ) * config.convdims))

            # We need to define a dummy array as a Keras train step need labels
            # (even if they are not used)
            dummy_Z = np.zeros(Znp.shape)
            dummy_samples = np.zeros(samples.shape)

            if ((epoch * config.epoch_iters + it) % (config.k + 1)) == 0:
                # Training the generator
                G_losses.append(DG.train_on_batch(Znp, dummy_Z))

            else:
                # Training the discriminator
                losses = Adv.train_on_batch([samples, Znp],
                                            [dummy_samples, dummy_Z])

                D_losses.append(losses[0])
                D_real_losses.append(losses[1])
                D_fake_losses.append(losses[2])

        # Collecting losses
        G_loss = float(np.mean(G_losses))
        D_loss = float(np.mean(D_losses))
        D_real_loss = float(np.mean(D_real_losses))
        D_fake_loss = float(np.mean(D_fake_losses))

        # Generating a sample image and saving it
        if main_thread and (generate_png or generate_hdf5):
            data = G.predict(z_sample)

        if generate_png and main_thread:
            out = np.squeeze((data + 1.) * 128.)
            image = Image.fromarray(np.uint8(out))
            image.save(samples_dir + run_name + "_" + str(epoch) + ".png")

        if generate_hdf5 and main_thread:
            f = h5py.File(
                samples_dir + run_name + "_" + str(epoch) + ".hdf5", mode="w")
            f.create_dataset('features', data=data)
            f.flush()
            f.close()

        # Logging
        if main_thread:
            print("Gcost=", G_loss, "Dcost=", D_loss, "Dreal_cost=", D_real_loss,
              "Dfake_cost=", D_fake_loss)

        if use_tensorboard and main_thread:
            # Logging losses
            losses_summary = tf.Summary(value=[
                tf.Summary.Value(tag="D_cost", simple_value=D_loss),
                tf.Summary.Value(tag="D_real_cost", simple_value=D_real_loss),
                tf.Summary.Value(tag="D_fake_cost", simple_value=D_fake_loss),
                tf.Summary.Value(tag="G_cost", simple_value=G_loss)
            ])

            # Logging samples
            data = np.transpose(data, (0, 2, 3, 1))
            sample_pl = tf.placeholder(
                tf.float32, shape=data.shape, name='img')
            with tf.Session() as sess:
                samples_summary = sess.run(
                    tf.summary.image(str(epoch), sample_pl),
                    feed_dict={
                        sample_pl: data
                    })

            writer.add_summary(losses_summary, global_step=epoch)
            writer.add_summary(samples_summary, global_step=epoch)

            writer.flush()

        # Saving
        if main_thread:
            G.save(checkpoints_dir + "/" + run_name + "_G_" + str(epoch) + ".hdf5")
            D.save(checkpoints_dir + "/" + run_name + "_D_" + str(epoch) + ".hdf5")
            DG.save(
                checkpoints_dir + "/" + run_name + "_DG_" + str(epoch) + ".hdf5")
            Adv.save(
                checkpoints_dir + "/" + run_name + "_Adv_" + str(epoch) + ".hdf5")

    # Closing the logger
    if use_tensorboard and main_thread:
        writer.close()
