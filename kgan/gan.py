from time import time
import sys

from keras.engine import Model

from kgan.factory import get_losses, get_optimizer, get_generator, get_discriminator

class TimePrint(object):
    '''
    Simple convenience class to print who long it takes between successive calls to its __init__ function.
    Usage example:
        TimePrint("some text")          -- simply prints "some text"
        <do some stuff here>
        TimePrint("some other text ")   -- prints "some other text (took ?s)", where ? is the time passed since TimePrint("some text") was called
    '''
    t_last = None

    def __init__(self, text):
        TimePrint.p(text)

    @classmethod
    def p(cls, text):
        t = time()
        print(text)
        if cls.t_last != None:
            print(" (took ", t - cls.t_last, "s)")
        cls.t_last = t
        sys.stdout.flush()


def gan(config):

    # Setting up
    loss_true, loss_fake, loss_gen = get_losses(config.losses)
    optimizer = get_optimizer(config.optimizer)(**config.optimizer_params)
    G = get_generator(config.generator)(config)
    D = get_discriminator(config.discriminator)(config)

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
