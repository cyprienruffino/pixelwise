import pickle
from io import TextIOWrapper
from config import Config


def load_models(config, D_path, G_path, DG_path, Adv_path):
    from kgan.gan import gan
    from keras.models import load_model

    if D_path is not None and G_path is not None and DG_path is not None and Adv_path is not None:
        custom_objects = {
            config.loss_true.__name__: config.loss_disc,
            config.loss_gen.__name__: config.loss_gen
        }

        G = load_model(G_path, custom_objects=custom_objects)
        D = load_model(D_path, custom_objects=custom_objects)
        DG = load_model(DG_path, custom_objects=custom_objects)
        Adv = load_model(Adv_path, custom_objects=custom_objects)
    else:
        optimizer = config.optimizer(config.optimizer_params)

        G = config.generator(config.zx,
                             convdims=config.convdims,
                             channes=config.nc,
                             gen_depth=config.gen_depth)

        D = config.discriminator(config.npx,
                                 convdims=config.convdims,
                                 channels=config.nc,
                                 disc_depth=config.disc_depth,
                                 clip_weights=config.clip_weights,
                                 clipping_value=config.c)

        if config.gradient_penalty:
            from kgan.losses import gradient_penalty
            gp = gradient_penalty(D.input, G.input, D, G)
            loss_disc_fake = lambda y_true, y_pred: config.loss_disc_fake(y_true, y_pred) + (config.lmbda * gp(y_true, y_pred)) / 2
            loss_disc_true = lambda y_true, y_pred: config.loss_disc_true(y_true, y_pred) + (config.lmbda * gp(y_true, y_pred)) / 2

        else:
            loss_disc_fake = config.loss_disc_fake
            loss_disc_true = config.loss_disc_true

        D, G, DG, Adv = gan(D, G, loss_disc_true, loss_disc_fake, config.loss_gen, optimizer)

    return D, G, DG, Adv


def load_config(sgancfg):
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
    return config
