from keras.engine import Model

from tools import TimePrint
from factory import get_losses, get_optimizer, get_generator, get_discriminator


def gan(config):

    # Setting up
    loss_true, loss_fake, loss_gen = get_losses(config.losses)
    optimizer = get_optimizer(config.optimizer)(*config.optimizer_params)
    G = get_generator(config.generator)
    D = get_discriminator(config.discriminator)

    # Creating and compiling the models
    TimePrint("Compiling the network...\n")

    X = D.input
    Z = G.input

    for layer in G.layers:
        layer.trainable = False

    Adv = Model(inputs=[X, Z], outputs=[D(X), D(G(Z))], name="Adv")

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
