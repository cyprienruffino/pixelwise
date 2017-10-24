import keras.backend as K


def softplus_gan_true(y_true, y_pred):
    return -K.mean(K.log(K.log(K.exp(K.flatten(y_pred)) + 1)))


def softplus_gan_fake(y_true, y_pred):
    return -K.mean(K.log(K.log(K.exp(1 - K.flatten(y_pred)) + 1)))


def softplus_gan_gen(y_true, y_pred):
    return softplus_gan_true(y_true, y_pred)


def gan_true(y_true, y_pred):
    return -K.mean(K.log(K.flatten(y_pred) + K.epsilon()))


def gan_fake(y_true, y_pred):
    return -K.mean(K.log(1 - K.flatten(y_pred) + K.epsilon()))


def gan_gen(y_true, y_pred):
    return gan_true(y_true, y_pred)


def wasserstein_true(y_true, y_pred):
    return K.mean(K.flatten(y_pred))


def wasserstein_fake(y_true, y_pred):
    return -K.mean(K.flatten(y_pred))


def wasserstein_gen(y_true, y_pred):
    return -K.mean(K.flatten(y_pred))
