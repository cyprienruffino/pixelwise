import pickle
import time

import h5py
import numpy as np
from tensorflow import set_random_seed

from tools import TimePrint


def train(sgancfg,
          training_image,
          run_name=None,
          checkpoints_dir="./",
          logs_dir="./",
          samples_dir="./",
          use_tensorboard=True,
          plot_models=True,
          D_path=None,
          G_path=None,
          DG_path=None,
          initial_epoch=0):

    if run_name is None:
        run_name = str(time.time())

    # Loading the config file
    config = pickle.load(sgancfg)

    # Seeding the random numbers generators
    np.random.seed(config.seed)
    set_random_seed(config.seed)

    # Importing Keras must be done after the seeding
    from sgan import sgan
    from losses import loss_d, loss_g
    from keras.optimizers import Adam
    from keras.models import load_model
    from keras.utils import plot_model
    from keras.backend import get_session

    # Load or create the model
    if D_path is not None and G_path is not None and DG_path is not None:
        G = load_model(G_path)
        D = load_model(D_path)
        DG = load_model(DG_path)
    else:
        D, G, DG = sgan(config)

    # Compiling the models (G don't need to be compiled)
    TimePrint("Compiling the network...\n")
    D.compile(optimizer=Adam(lr=config.lr, beta_1=config.b1), loss=loss_d)
    TimePrint("Discriminator done.")
    DG.compile(optimizer=Adam(lr=config.lr, beta_1=config.b1), loss=loss_g)
    TimePrint("Generator done.")

    # Setting up the TensorBoard logger
    if use_tensorboard:
        import tensorflow as tf
        writer = tf.summary.FileWriter(logs_dir + "/" + run_name + ".tblog",
                                       get_session().graph)

    # Plotting the models
    if plot_models:
        plot_model(D, logs_dir + "/D.png")
        plot_model(G, logs_dir + "/G.png")
        plot_model(DG, logs_dir + "/DG.png")

    # Do the actual training
    z_sample = np.random.uniform(-1., 1., (1, config.nz) +
                                 ((config.zx_sample, ) * config.convdims))

    for epoch in range(config.epochs):
        print("Epoch", epoch)

        for it in range(config.epoch_iters):
            samples = next(data)

            G_losses = []
            D_losses = []

            # Creating the input noise
            Znp = np.random.uniform(-1., 1., (config.batch_size, config.nz) +
                                    ((config.zx, ) * config.convdims))

            # We need to define a dummy array as a Keras train step need labels
            # (even if they are not used)
            dummy_Z = np.zeros(Znp.shape)
            dummy_samples = np.zeros(samples.shape)

            if (config.epochs * config.epoch_iters + it) % (config.k + 1) == 0:
                # Training the generator
                # We need to freeze the discriminator
                D.trainable = False
                G_losses.append(DG.train_on_batch(Znp, dummy_Z))
                D.trainable = True
            else:
                # Training the discriminator
                D_losses.append(DG.train_on_batch(samples, dummy_samples))

            G_loss = np.mean(G_losses)
            D_loss = np.mean(D_losses)

            # Logging
            print("Gcost=", G_loss, "  Dcost=", D_loss)
            if use_tensorboard:
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="D_cost", simple_value=D_loss),
                    tf.Summary.Value(tag="G_cost", simple_value=G_loss)
                ])
                writer.add_summary(summary)
                writer.flush()

            # Generating a sample image and saving it
            data = G.predict(z_sample)
            f = h5py.File(
                samples_dir + run_name + "_" + str(epoch) + ".hdf5", mode="w")
            f.create_dataset('features', data=data)
            f.flush()
            f.close()

            # Saving
            G.save(checkpoints_dir + "/" + run_name + "_G_" + str(epoch) +
                   ".hdf5")
            D.save(checkpoints_dir + "/" + run_name + "_D_" + str(epoch) +
                   ".hdf5")
            DG.save(checkpoints_dir + "/" + run_name + "_DG_" + str(epoch) +
                    ".hdf5")

    # Closing the logger
    if use_tensorboard:
        writer.close()
