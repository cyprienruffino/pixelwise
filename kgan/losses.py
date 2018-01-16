def softplus_gan_disc(y_true, y_pred):
    import keras.backend as K
    return (-K.mean(K.log(K.log(K.exp(K.flatten(y_pred[0])) + 1)))
        -K.mean(K.log(K.log(K.exp(1 - K.flatten(y_pred[1])) + 1))))


def softplus_gan_gen(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.log(K.log(K.exp(K.flatten(y_pred)) + 1)))


def gan_disc(y_true, y_pred):
    import keras.backend as K
    return (-K.mean(K.log(K.flatten(y_pred[0])))
        -K.mean(K.log(1 - K.flatten(y_pred[1]))))


def gan_gen(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.log(K.flatten(y_pred[0])))


def epsilon_gan_disc(y_true, y_pred):
    import keras.backend as K
    return (-K.mean(K.log(K.flatten(y_pred[0]) + K.epsilon()))
        -K.mean(K.log(1 - K.flatten(y_pred[1]) + K.epsilon())))


def epsilon_gan_gen(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.log(K.flatten(y_pred) + K.epsilon()))


def wasserstein_disc(y_true, y_pred):
    import keras.backend as K
    return K.mean(K.flatten(y_pred[1])) - K.mean(K.flatten(y_pred[0]))


def wasserstein_gen(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.flatten(y_pred))


def wasserstein_min_disc(y_true, y_pred):
    import keras.backend as K
    return K.min(K.flatten(y_pred[1])) - K.min(K.flatten(y_pred[0]))


def wasserstein_min_gen(y_true, y_pred):
    import keras.backend as K
    return -K.min(K.flatten(y_pred))


