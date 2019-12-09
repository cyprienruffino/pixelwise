from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import regularizers as kr

from layers import InstanceNormalization


def ResidualBlock(filters, nb_layers=2, kernel_size=3, normalization="batchnorm"):
    def _resblock(inp):
        layer = inp
        for i in range(nb_layers - 1):
            layer = kl.Conv2D(filters, kernel_size=kernel_size,
                           padding="same", kernel_regularizer=kr.l2())(layer)

            if normalization is "batchnorm":
                layer = kl.BatchNormalization()(layer)
            elif normalization is "instancenorm":
                layer = InstanceNormalization()(layer)
            layer = kl.Activation("relu")(layer)

        layer = kl.Conv2D(filters, kernel_size=kernel_size,
                       padding="same", kernel_regularizer=kr.l2())(layer)
        if normalization is "batchnorm":
            layer = kl.BatchNormalization()(layer)
        elif normalization is "instancenorm":
            layer = InstanceNormalization()(layer)

        return kl.add([layer, inp])
    return _resblock



