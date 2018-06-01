import numpy as np
import progressbar

import utils


def train(sgancfg,
          run_name,
          disc_data_provider,
          gen_data_provider,
          checkpoints_dir="./",
          logs_dir="./",
          samples_dir="./",
          use_tensorboard=True,
          use_matplotlib=False,
          checkpoint_models=True,
          plot_models=False,
          save_json=True,
          generate_png=True,
          generate_hdf5=True,
          save_summaries=True,
          D_path=None,
          G_path=None):
    utils.create_dirs(logs_dir, checkpoints_dir, samples_dir)
    config = utils.load_config(sgancfg)

    # Seeding the random numbers generators
    np.random.seed(config.seed)

    # Load or create the model
    D, G, DG, Adv = utils.load_models(config, D_path, G_path)

    # Setting up the TensorBoard logger
    if use_tensorboard:
        writer = utils.setup_tensorboard(logs_dir, run_name)

    if plot_models: utils.plot_models(D, G, DG, Adv, logs_dir)
    if save_summaries: utils.save_summaries(D, G, DG, Adv, logs_dir)
    if save_json: utils.save_jsons(D, G, DG, Adv, logs_dir)

    # Do the actual training
    G_losses_history = []
    D_losses_history = []
    for epoch in range(config.epochs):
        print("Epoch", epoch)
        iters = config.epoch_iters // config.k
        bar = progressbar.ProgressBar(maxvalue=iters)

        G_losses = []
        D_losses = []
        for _ in bar(range(iters)):
            Znp = next(gen_data_provider)

        # We need to define a dummy array as a Keras train step need labels
        # (even if they are not used)
            dummy_Z = np.zeros(Znp.shape)

            # Training the generator
            losses = DG.train_on_batch(Znp, dummy_Z)
            G_losses.append(losses)

            # Training the discriminator
            for _ in range(config.k):
                Znp =  next(gen_data_provider)
                samples = next(disc_data_provider)
                losses = Adv.train_on_batch([samples, Znp], [dummy_Z, dummy_Z])
                D_losses.append(losses[0] + losses[1])

        # Epoch end, logging
        G_loss = float(np.mean(G_losses))
        D_loss = float(np.mean(D_losses))
        G_losses_history.append(G_loss)
        D_losses_history.append(D_loss)

        if generate_png or generate_hdf5 or use_tensorboard:
            data = G.predict(np.random.uniform(-1, 1, (1, 1, config.zx_sample, config.zx_sample)))

        if config.convdims == 2:
            if generate_png: utils.gen_png(data, samples_dir, run_name, epoch)
            if use_tensorboard: utils.tensorboard_log_image(data, writer, epoch)

        if generate_hdf5: utils.gen_hdf5(data, samples_dir, run_name, epoch)
        if use_tensorboard: utils.tensorboard_log_losses(D_loss, G_loss, writer, epoch)
        if checkpoint_models: utils.save_models(D, G, checkpoints_dir, run_name, epoch)

    # Run end
    if use_tensorboard: writer.close()
    if use_matplotlib: utils.plot_losses(D_losses_history, G_losses_history, run_name)
