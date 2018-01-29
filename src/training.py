import time
import progressbar
import shutil

import numpy as np
from tensorflow import set_random_seed

import log
import utils


def sample_noise(config):
    return np.random.uniform(-1., 1., (config.batch_size, config.nz) +
                             ((config.zx, ) * config.convdims))


def generate_sample(G, config):
    z_sample = np.random.uniform(-1., 1., (1, config.nz) +
                                 ((config.zx_sample, ) * config.convdims))
    dummy_Z = np.zeros(z_sample.shape)
    return G.predict(z_sample, dummy_Z)


def train(sgancfg,
          data_provider,
          run_name=None,
          checkpoints_dir="./",
          logs_dir="./",
          samples_dir="./",
          progress_bar=True,
          use_tensorboard=True,
          use_matplotlib=True,
          checkpoint_models=True,
          plot_models=True,
          save_json=True,
          save_config_file=True,
          generate_png=True,
          generate_hdf5=True,
          log_metadata=True,
          D_path=None,
          G_path=None,
          DG_path=None,
          Adv_path=None,
          initial_epoch=0):

    if run_name is None:
        run_name = str(time.time())

    log.create_dirs(logs_dir, checkpoints_dir, samples_dir)
    config = utils.load_config(sgancfg)

    # Seeding the random numbers generators
    np.random.seed(config.seed)
    set_random_seed(config.seed)

    # Load or create the model
    D, G, DG, Adv = utils.load_models(config, D_path, G_path, DG_path, Adv_path)

    # Setting up the TensorBoard logger
    if use_tensorboard:
        writer = log.setup_tensorboard(logs_dir, run_name)

    if plot_models: log.plot_models(D, G, DG, Adv, logs_dir)
    if save_json: log.save_jsons(D, G, DG, Adv, logs_dir)

    # Do the actual training
    G_losses_history = []
    D_losses_history = []
    for epoch in range(config.epochs):
        print("Epoch", epoch)
        bar = progressbar.ProgressBar(maxval=int(config.epoch_iters / config.batch_size))

        G_losses = []
        D_losses = []
        for it in bar(range(int(config.epoch_iters / config.batch_size))):
            samples = next(data_provider)

            Znp = sample_noise(config)

            # We need to define a dummy array as a Keras train step need labels
            # (even if they are not used)
            dummy_Z = np.zeros(Znp.shape)

            # Training the generator
            losses = DG.train_on_batch(Znp, dummy_Z)
            G_losses.append(losses)

            # Training the discriminator
            for _ in range(config.k):
                losses = Adv.train_on_batch([samples, Znp], dummy_Z)
                D_losses.append(losses)

        # Epoch end, logging
        G_loss = float(np.mean(G_losses))
        D_loss = float(np.mean(D_losses))
        G_losses_history.append(G_loss)
        D_losses_history.append(D_loss)
        print("Gcost=", G_loss, "Dcost=", D_loss)

        if generate_png or generate_hdf5 or use_tensorboard: data = generate_sample(G)
        if generate_png: log.gen_png(data, samples_dir, run_name, epoch)
        if generate_hdf5: log.gen_hdf5(data, samples_dir, run_name, epoch)
        if use_tensorboard: log.tensorboard_log(data, writer, epoch)
        if checkpoint_models: log.save_models(D, G, DG, Adv, checkpoints_dir, epoch)

    # Run end
    if use_tensorboard: writer.close()
    if use_matplotlib: log.plot_losses(D_losses_history, G_losses_history, run_name)
