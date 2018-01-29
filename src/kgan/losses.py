def softplus_gan_disc(y_true, y_pred):
    import keras.backend as K
    return (-K.mean(K.log(K.log(K.exp(K.flatten(y_pred[0])) + 1)))
        -K.mean(K.log(K.log(K.exp(1 - K.flatten(y_pred[1])) + 1))))


def softplus_gan_gen(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.log(K.log(K.exp(K.flatten(y_pred)) + 1)))


def gan_disc(y_true, y_pred):
    import keras.backend as K
    return (-K.mean(K.log(K.flatten(y_pred[0])))
        -K.mean(K.log(1 - K.flatten(y_pred[1]))))


def gan_gen(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.log(K.flatten(y_pred[0])))


def epsilon_gan_disc(y_true, y_pred):
    import keras.backend as K
    return (-K.mean(K.log(K.flatten(y_pred[0]) + K.epsilon()))
        -K.mean(K.log(1 - K.flatten(y_pred[1]) + K.epsilon())))


def epsilon_gan_gen(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.log(K.flatten(y_pred) + K.epsilon()))


def wasserstein_disc(y_true, y_pred):
    import keras.backend as K
    return K.mean(K.flatten(y_pred[1])) - K.mean(K.flatten(y_pred[0]))


def wasserstein_gen(y_true, y_pred):
    import keras.backend as K
    return -K.mean(K.flatten(y_pred))


def wasserstein_min_disc(y_true, y_pred):
    import keras.backend as K
    return K.min(K.flatten(y_pred[1])) - K.min(K.flatten(y_pred[0]))


def wasserstein_min_gen(y_true, y_pred):
    import keras.backend as K
    return -K.min(K.flatten(y_pred))


def gradient_penalty(X, Z, D, G):
    def _gp(y_true, y_pred):
        import keras.backend as K

        # Sampling points on the line between a true and a fake data
        epsilon = K.random_uniform((K.shape(X)[0], 1, 1, 1))
        samples = epsilon * X + (1 - epsilon) * G(Z)

        # Computing the discriminator gradient on the points
        grads = K.gradients(K.mean(D(samples)), samples)

        # Computing the gradient norm
        norm = K.sqrt(K.sum(K.square(grads)))

        # Computing the actual penalty
        penalty = K.square(norm - 1)

        return penalty

    return _gp