import keras.backend as K


def gan_disc_fake(y_true, y_pred):
    return -K.mean(K.log(1 - y_pred + K.epsilon()))


def gan_disc_true(y_true, y_pred):
    return -K.mean(K.log(y_pred + K.epsilon()))


def gan_gen(y_true, y_pred):
    return -K.mean(K.log(y_pred + K.epsilon()))
