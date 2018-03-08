def gan_disc_fake(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.log(1 - K.flatten(y_pred) + K.epsilon()))


def gan_disc_true(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.log(K.flatten(y_pred) + K.epsilon()))


def gan_gen(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.log(K.flatten(y_pred) + K.epsilon()))
