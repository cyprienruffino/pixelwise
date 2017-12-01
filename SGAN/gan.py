from keras.engine import Model
from keras.optimizers import SGD, RMSprop, Adam

from generator import create_gen
from discriminator import create_disc
from config import Optimizer, Losses
from tools import TimePrint


def gan(config):
    # Creating the generator and discriminator
    D = create_disc(config)
    X = D.input

    G = create_gen(config)
    Z = G.input

    # Selecting the losses
    if config.losses == Losses.classical_gan:
        from losses import gan_true as loss_true
        from losses import gan_fake as loss_fake
        from losses import gan_gen as loss_gen
    elif config.losses == Losses.epsilon_gan:
        from losses import epsilon_gan_true as loss_true
        from losses import epsilon_gan_fake as loss_fake
        from losses import epsilon_gan_gen as loss_gen
    elif config.losses == Losses.wasserstein_gan:
        from losses import wasserstein_true as loss_true
        from losses import wasserstein_fake as loss_fake
        from losses import wasserstein_gen as loss_gen
    elif config.losses == Losses.wasserstein_min_gan:
        from losses import wasserstein_min_true as loss_true
        from losses import wasserstein_min_fake as loss_fake
        from losses import wasserstein_min_gen as loss_gen
    elif config.losses == Losses.softplus_gan:
        from losses import softplus_gan_true as loss_true
        from losses import softplus_gan_fake as loss_fake
        from losses import softplus_gan_gen as loss_gen
    else:
        raise "Unknown losses " + config.losses

    # Creating and compiling the models
    TimePrint("Compiling the network...\n")

    for layer in G.layers:
        layer.trainable = False

    Adv = Model(inputs=[X, Z], outputs=[D(X), D(G(Z))], name="Adv")

    # Selecting the optimizer
    if config.optimizer == Optimizer.adam:
        optimizer = Adam(lr=config.lr, beta_1=config.b1)
    elif config.optimizer == Optimizer.rmsprop:
        optimizer = RMSprop(lr=config.lr)
    elif config.optimizer == Optimizer.sgd:
        optimizer = SGD(lr=config.lr, momentum=config.momentum)
    else:
        raise "Unknown optimizer " + config.optimizer

    Adv.compile(
        optimizer=optimizer, loss=[loss_true, loss_fake], loss_weights=[1, 1])

    TimePrint("Discriminator done.")

    for layer in D.layers:
        layer.trainable = False
    for layer in G.layers:
        layer.trainable = True

    DG = Model(inputs=Z, outputs=D(G(Z)), name="DG")
    DG.compile(optimizer=optimizer, loss=loss_gen)

    TimePrint("Generator done.")

    for layer in D.layers:
        layer.trainable = True

    return D, G, DG, Adv
