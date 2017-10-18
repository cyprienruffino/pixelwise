import keras.backend as K


def gan_true(y_true, y_pred):
    return -K.mean(K.log(K.flatten(y_pred) + K.epsilon()))


def gan_fake(y_true, y_pred):
    return -K.mean(K.log(1 - K.flatten(y_pred) + K.epsilon()))


def gan_gen(y_true, y_pred):
    return -K.mean(K.log(K.flatten(y_pred) + K.epsilon()))


def wasserstein_true(y_true, y_pred):
    return K.mean(K.flatten(y_pred))


def wasserstein_false(y_true, y_pred):
    return -K.mean(K.flatten(y_pred))


def wasserstein_gen(y_true, y_pred):
    return -K.mean(K.flatten(y_pred))
