import keras.backend as K


def wasserstein_disc_fake(y_true, y_pred):
    return K.mean(y_pred)


def wasserstein_disc_true(y_true, y_pred):
    return -K.mean(y_pred)


def wasserstein_gen(y_true, y_pred):
    return -K.mean(y_pred)


def gradient_penalty(X, Z, D, G):
    def _gp(y_true, y_pred):

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
