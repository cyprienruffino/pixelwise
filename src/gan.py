def gan(discriminator, generator, loss_disc_true, loss_disc_fake, loss_gen, disc_optimizer, gen_optimizer):
    from keras.engine import Model

    X = discriminator.input
    Z = generator.input

    Xfake = generator(Z)

    if type(X) is not list:
        X = [X]
    if type(Z) is not list:
        Z = [Z]

    if type(Xfake) is not list:
        Xfake = [Xfake]

    for layer in discriminator.layers:
        layer.trainable = False

    DG = Model(inputs=Z + X[1:], outputs=[discriminator(Xfake + X[1:])] + Xfake[1:], name="DG")
    DG.compile(optimizer=gen_optimizer, loss=loss_gen)

    for layer in generator.layers:
        layer.trainable = False
    for layer in discriminator.layers:
        layer.trainable = True

    Adv = Model(inputs=X + Z, outputs=[discriminator(X)] + [discriminator(Xfake + X[1:])] + Xfake[1:], name="Adv")
    Adv.compile(optimizer=disc_optimizer, loss=[loss_disc_true, loss_disc_fake])

    return discriminator, generator, DG, Adv
