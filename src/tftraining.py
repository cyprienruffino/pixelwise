import numpy as np
import progressbar
from tensorflow import set_random_seed
import tensorflow as tf
import keras.backend as K

import utils


def gan(discriminator, generator):

    Z = generator.input
    X = discriminator.input

    Gen = generator(Z)
    Dreal = discriminator(X)
    Dfake = discriminator(Gen)

    dloss = - K.mean(K.log(1 - Dfake + K.epsilon())) - K.mean(K.log(Dreal + K.epsilon()))
    gloss = - K.mean(K.log(Dfake + K.epsilon()))

    Goptimizer = tf.train.AdamOptimizer(0.0005, beta1=0.5)
    Doptimizer = tf.train.AdamOptimizer(0.0005, beta1=0.5)

    grad_loss_wd = Doptimizer.compute_gradients(dloss, discriminator.trainable_weights)
    update_D = Doptimizer.apply_gradients(grad_loss_wd)

    grad_loss_wg = Goptimizer.compute_gradients(gloss, generator.trainable_weights)
    update_G = Goptimizer.apply_gradients(grad_loss_wg)

    def get_internal_updates(model):
        # get all internal update ops (like moving averages) of a model
        inbound_nodes = model.inbound_nodes
        input_tensors = []
        for ibn in inbound_nodes:
            input_tensors += ibn.input_tensors
        updates = [model.get_updates_for(i) for i in input_tensors]
        return updates

    other_parameter_updates = [get_internal_updates(m) for m in [discriminator, generator]]

    return [update_D, update_G, other_parameter_updates], [dloss, gloss], discriminator, generator, X, Z


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
    set_random_seed(config.seed)

    sess = K.get_session()

    train_step, losses, D, G, X, Z = gan(config.discriminator(**config.disc_args), config.generator(**config.gen_args))

    # Setting up the TensorBoard logger
    if use_tensorboard:
        writer = utils.setup_tensorboard(logs_dir, run_name)

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
            samples = next(disc_data_provider)
            if len(samples) is config.batch_size:
                loss = sess.run([train_step, losses], feed_dict={Z: Znp, X: samples, K.learning_phase(): True})[1]
                D_losses.append(loss[0])
                G_losses.append(loss[1])

        # Logging
        prediction = G.predict(np.random.uniform(-1, 1, (1, 1, config.zx, config.zx)))

        if use_tensorboard:
            utils.tensorboard_log_losses(np.mean(D_losses), np.mean(G_losses), writer, epoch)
            utils.tensorboard_log_image(prediction, writer, epoch)

        utils.gen_png(prediction, samples_dir, run_name, epoch)

        if generate_hdf5:
            utils.gen_hdf5(prediction, samples_dir, run_name, epoch)
        if checkpoint_models:
            utils.save_models(D, G, checkpoints_dir, run_name, epoch)

    # Run end
    if use_tensorboard:
        writer.close()
    if use_matplotlib:
        utils.plot_losses(D_losses_history, G_losses_history, run_name)
