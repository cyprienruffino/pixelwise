
def load_models(config, D_path, G_path):
    from gan import gan
    from keras.models import load_model

    if D_path is not None and G_path is not None:
        custom_objects = {
            config.loss_disc_true.__name__: config.loss_disc_true,
            config.loss_disc_fake.__name__: config.loss_disc_fake,
            config.loss_gen.__name__: config.loss_gen
        }
        G = load_model(G_path, custom_objects=custom_objects)
        D = load_model(D_path, custom_objects=custom_objects)

    else:
        G = config.generator(**config.gen_args)
        D = config.discriminator(**config.disc_args)

    D_optimizer = config.disc_optimizer(**config.disc_optimizer_args)
    G_optimizer = config.gen_optimizer(**config.gen_optimizer_args)

    D, G, DG, Adv = gan(D, G, config.loss_disc_true, config.loss_disc_fake, config.loss_gen, D_optimizer, G_optimizer)

    return D, G, DG, Adv


def save_models(D, G, checkpoints_dir, run_name, epoch):
    G.save(checkpoints_dir + "/" + run_name + "_G_" + str(epoch) + ".hdf5")
    D.save(checkpoints_dir + "/" + run_name + "_D_" + str(epoch) + ".hdf5")
