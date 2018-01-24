def Adam(args):
    from keras.optimizers import Adam
    return Adam(**args)


def SGD(args):
    from keras.optimizers import SGD
    return SGD(**args)


def RMSProp(args):
    from keras.optimizers import RMSprop
    return RMSprop(**args)


def Adagrad(args):
    from keras.optimizers import Adagrad
    return Adagrad(**args)
