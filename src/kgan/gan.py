def gan(discriminator, generator, loss_disc, loss_gen, optimizer):
    from keras.engine import Model
    from keras.layers import concatenate

    X = discriminator.input
    Z = generator.input

    dg = discriminator(generator(Z))

    for layer in generator.layers:
        layer.trainable = False

    adv_output = concatenate([discriminator(X), dg], axis=0)
    Adv = Model(inputs=[X, Z], outputs=adv_output, name="Adv")
    Adv.compile(optimizer=optimizer, loss=loss_disc)

    for layer in discriminator.layers:
        layer.trainable = False
    for layer in generator.layers:
        layer.trainable = True

    DG = Model(inputs=Z, outputs=dg, name="DG")
    DG.compile(optimizer=optimizer, loss=loss_gen)

    return discriminator, generator, DG, Adv
