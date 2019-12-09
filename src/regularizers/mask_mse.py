import tensorflow as tf


def mask_mse(D, G, Z, X, C, Cf):
    mask = tf.cast(tf.greater(tf.abs(Cf), 0), tf.float32)  # Getting binary mask of Cf
    nb_const = tf.reduce_sum(mask)
    masked_image = mask * G.output
    return tf.reduce_sum(tf.square(Cf - masked_image)) / nb_const
