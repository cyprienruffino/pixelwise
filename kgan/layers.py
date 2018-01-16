from keras.layers import Layer, InputSpec
from keras import initializers, regularizers
import keras.backend as K


def to_list(x):
    if type(x) not in [list, tuple]:
        return [x]
    else:
        return list(x)


def LN(x, gamma, beta, epsilon=1e-6, axis=-1):
    m = K.mean(x, axis=axis, keepdims=True)
    std = K.sqrt(K.var(x, axis=axis, keepdims=True) + epsilon)
    x_normed = (x - m) / (std + epsilon)
    x_normed = gamma * x_normed + beta

    return x_normed


class LayerNormalization(Layer):
    def __init__(self, axis=-1,
                 gamma_init='one', beta_init='zero',
                 gamma_regularizer=None, beta_regularizer=None,
                 epsilon=1e-6, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)

        self.axis = to_list(axis)
        self.gamma_init = initializers.get(gamma_init)
        self.beta_init = initializers.get(beta_init)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.epsilon = epsilon

        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = [1 for _ in input_shape]
        for i in self.axis:
            shape[i] = input_shape[i]
        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='beta')
        self.built = True

    def call(self, inputs, mask=None):
        return LN(inputs, gamma=self.gamma, beta=self.beta,
                  axis=self.axis, epsilon=self.epsilon)

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_init': initializers.serialize(self.gamma_init),
                  'beta_init': initializers.serialize(self.beta_init),
                  'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'beta_regularizer': regularizers.serialize(self.gamma_regularizer)}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
