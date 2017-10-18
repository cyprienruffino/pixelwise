import keras.backend as K


def loss_true(y_true, y_pred):
    return -K.mean(K.log(K.flatten(y_pred) + K.epsilon()))


def loss_fake(y_true, y_pred):
    return -K.mean(K.log(1 - K.flatten(y_pred) + K.epsilon()))
