import tensorflow as tf
from tensorflow.python.keras import layers as kl


def ConvCoord2D():
    def __coordconv2D(inp):
        stacked = tf.to_float(
            tf.stack([tf.expand_dims(tf.range(0, inp.shape.as_list()[1]), axis=-1)] * inp.shape.as_list()[2], axis=1))
        xs = tf.tile(tf.expand_dims(stacked, axis=0), tf.stack([tf.shape(inp)[0], 1, 1, 1]))

        stacked = tf.to_float(tf.stack(
            [tf.transpose(tf.expand_dims(tf.range(0, inp.shape.as_list()[2]), axis=0))] * inp.shape.as_list()[1]))
        ys = tf.tile(tf.expand_dims(stacked, axis=0), tf.stack([tf.shape(inp)[0], 1, 1, 1]))
        return tf.concat([inp, xs, ys], axis=-1)

    return kl.Lambda(__coordconv2D)
