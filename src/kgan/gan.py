def gan(discriminator, generator, loss_disc_true, loss_disc_fake, loss_gen, optimizer):
    from keras.engine import Model

    X = discriminator.input
    Z = generator.input

    for layer in generator.layers:
        layer.trainable = False

    Adv = Model(inputs=[X, Z], outputs=[discriminator(X), discriminator(generator(Z))], name="Adv")
    Adv.compile(optimizer=optimizer, loss=[loss_disc_true, loss_disc_fake])

    for layer in discriminator.layers:
        layer.trainable = False
    for layer in generator.layers:
        layer.trainable = True

    DG = Model(inputs=Z, outputs=discriminator(generator(Z)), name="DG")
    DG.compile(optimizer=optimizer, loss=loss_gen)

    return discriminator, generator, DG, Adv
