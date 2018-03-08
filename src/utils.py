import pickle
from io import TextIOWrapper
from config import Config


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

    optimizer = config.optimizer(**config.optimizer_params)
    D, G, DG, Adv = gan(D, G, config.loss_disc_true, config.loss_disc_fake, config.loss_gen, optimizer)

    return D, G, DG, Adv


def save_models(D, G, checkpoints_dir, run_name, epoch):
    G.save(checkpoints_dir + "/" + run_name + "_G_" + str(epoch) + ".hdf5")
    D.save(checkpoints_dir + "/" + run_name + "_D_" + str(epoch) + ".hdf5")


def load_config(sgancfg):
    # Loading the config file
    if type(sgancfg) == str or type(sgancfg) == TextIOWrapper:
        with open(sgancfg, "rb") as f:
            config = pickle.load(f)
    elif issubclass(type(sgancfg), Config):
        config = sgancfg
    else:
        raise TypeError(
            "sgancfg : unknown type. Must pass a path as a string, an opened file or a Config object"
        )
    return config
