import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers as kl


def tf_sqrtm(mat, eps=1e-10):
    # WARNING : This only works for symmetric matrices !
    s, u, v = tf.svd(mat)
    si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
    return tf.matmul(tf.matmul(u, tf.diag(si)), v, transpose_b=True)

def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx


def get_fid_funtion(model_path, data_shape, sess):
    """
    Build a model that computes the Fréchet Inception Distance
    :param model_path: Path to a pre-trained Keras model
    :param sess: The current TensorFlow session
    :return: A function fid(X_real, X_fake)
    """

    with tf.name_scope("FID"):
        cl = k.models.load_model(model_path)
        pl_real = tf.placeholder(tf.float32, data_shape)
        pl_fake = tf.placeholder(tf.float32, data_shape)

        acts_real = k.Model(cl.input, cl.layers[-4].output)(kl.Input(tensor=pl_real))
        acts_fake = k.Model(cl.input, cl.layers[-4].output)(kl.Input(tensor=pl_fake))

        mu_real = tf.reduce_mean(acts_real, axis=0)
        mu_fake = tf.reduce_mean(acts_fake, axis=0)
        sigma_real = tf_cov(acts_real)
        sigma_fake = tf_cov(acts_fake)

        diff = mu_real - mu_fake
        mu2 = tf.reduce_sum(tf.multiply(diff, diff))

        # Computing the trace of sqrt(sigma_real * sigma_fake)
        # Only works for the trace !
        # See https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
        sqrt_sigma = tf_sqrtm(sigma_real)
        sqrt_a_sigma_a = tf.matmul(
            sqrt_sigma, tf.matmul(sigma_fake, sqrt_sigma))

        tr = tf.trace(sigma_real + sigma_fake - 2 * tf_sqrtm(sqrt_a_sigma_a))
        fid = mu2 + tr

    def _fid(Xreal, Xfake):
        return sess.run(fid, feed_dict={pl_real: Xreal, pl_fake: Xfake})

    return _fid
