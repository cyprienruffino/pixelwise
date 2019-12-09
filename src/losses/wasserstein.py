import tensorflow as tf


def gradient_penalty(D, G, Z, X, batch_size):
    epsilon = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    X_hat = X + epsilon * (G.output - X)
    D_hat = D(X_hat)
    grad_D_hat = tf.gradients(D_hat, [X_hat])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_hat)))
    return tf.reduce_mean((slopes - 1.) ** 2)


def d_fake(D, G, Z, X):
    return tf.reduce_mean(D.output)


def d_real(D, G, Z, X):
    return - tf.reduce_mean(D[G.output])


def g(D, G, Z, X):
    return tf.reduce_mean(D(G.output))


def gradient_penalty_cond(D, G, Z, X, C, Cf, batch_size):
    epsilon = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    X_hat = X + epsilon * (G.output - X)
    D_hat = D([X_hat, C])
    grad_D_hat = tf.gradients(D_hat, [X_hat])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_hat)))
    return tf.reduce_mean((slopes - 1.) ** 2)


def d_fake_cond(D, G, Z, X, C, Cf):
    return tf.reduce_mean(D.output)


def d_real_cond(D, G, Z, X, C, Cf):
    return - tf.reduce_mean(D([G.output, Cf]))


def g_cond(D, G, Z, X, C, Cf):
    return tf.reduce_mean(D[G.output, Cf])
