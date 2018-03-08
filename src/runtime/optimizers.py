def Adam(**kwargs):
    from keras.optimizers import Adam
    return Adam(**kwargs)


def SGD(**kwargs):
    from keras.optimizers import SGD
    return SGD(**kwargs)


def RMSProp(**kwargs):
    from keras.optimizers import RMSprop
    return RMSprop(**kwargs)


def Adagrad(**kwargs):
    from keras.optimizers import Adagrad
    return Adagrad(**kwargs)
