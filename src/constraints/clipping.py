import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class Clip(Constraint):
    '''
    Clips the weights incident to each hidden unit to be inside a range
    Exactly the same as the keras_contrib one, but reimplemented to avoid a nearly pointless dependency
    '''

    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, p):
        return tf.clip_by_value(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__, 'c': self.c}
