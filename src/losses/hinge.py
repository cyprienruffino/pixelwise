import tensorflow as tf


def d_real(D, G, Z, X):
    return tf.reduce_mean(-tf.log(D.output + 1e-8))


def d_fake(D, G, Z, X):
    return tf.reduce_mean(-tf.log(1 - D(G.output) + 1e-8))


def g(D, G, Z, X):
    return tf.reduce_mean(-tf.log(D(G.output) + 1e-8))


def d_real_cond(D, G, Z, X, C, Cf):
    return tf.reduce_mean(-tf.log(D.output + 1e-8))


def d_fake_cond(D, G, Z, X, C, Cf):
    return tf.reduce_mean(-tf.log(1 - D(G.output) + 1e-8))


def g_cond(D, G, Z, X, C, Cf):
    return tf.reduce_mean(-tf.log(D([G.output, Cf]) + 1e-8))
