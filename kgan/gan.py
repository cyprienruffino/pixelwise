def gan(discriminator, generator, loss_true, loss_fake, loss_gen, optimizer):
    from keras.engine import Model

    X = discriminator.input
    Z = generator.input

    for layer in generator.layers:
        layer.trainable = False

    Adv = Model(
        inputs=[X, Z],
        outputs=[discriminator(X),
                 discriminator(generator(Z))],
        name="Adv")

    Adv.compile(
        optimizer=optimizer, loss=[loss_true, loss_fake], loss_weights=[1, 1])

    for layer in discriminator.layers:
        layer.trainable = False
    for layer in generator.layers:
        layer.trainable = True

    DG = Model(inputs=Z, outputs=discriminator(generator(Z)), name="DG")
    DG.compile(optimizer=optimizer, loss=loss_gen)

    for layer in discriminator.layers:
        layer.trainable = True

    return discriminator, generator, DG, Adv
