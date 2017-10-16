import keras.backend as K


def loss_X(y_true, y_pred):
    return -K.mean(K.log(K.flatten(y_pred)))


def loss_Z(y_true, y_pred):
    return -K.mean(K.log(1 - K.flatten(y_pred)))
