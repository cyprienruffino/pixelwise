import keras.backend as K


def loss_d(y_true, y_pred):
    return -K.mean(K.log(y_pred[0])) - K.mean(K.log(1 - y_pred[1]))


def loss_g(y_true, y_pred):
    return -K.mean(K.log(y_pred))
