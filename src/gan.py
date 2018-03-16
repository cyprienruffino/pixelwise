def gan(discriminator, generator, loss_disc_true, loss_disc_fake, loss_gen, optimizer):
    from keras.engine import Model

    X = discriminator.input
    Z = generator.input

    if type(X) is not list:
        X = [X]
    if type(Z) is not list:
        Z = [Z]

    for layer in discriminator.layers:
        layer.trainable = False

    DG = Model(inputs=Z, outputs=discriminator(generator(Z)), name="DG")
    DG.compile(optimizer=optimizer, loss=loss_gen)

    for layer in generator.layers:
        layer.trainable = False
    for layer in discriminator.layers:
        layer.trainable = True

    Adv = Model(inputs=X + Z, outputs=[discriminator(X), discriminator(generator(Z))], name="Adv")
    Adv.compile(optimizer=optimizer, loss=[loss_disc_true, loss_disc_fake])

    return discriminator, generator, DG, Adv
