import numpy as np
import progressbar
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers as kl


def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx


def tf_sqrtm(mat, eps=1e-10):
    # WARNING : This only works for symmetric matrices !
    s, u, v = tf.svd(mat)
    si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
    return tf.matmul(tf.matmul(u, tf.diag(si)), v, transpose_b=True)


def create_fid_func(model_path, sess, batch_size=1):
    with tf.name_scope("FID"):
        acts_real_pl = tf.placeholder(tf.float32, cl.layers[-2].output.shape)
        acts_fake_pl = tf.placeholder(tf.float32, cl.layers[-2].output.shape)

        mu_real = tf.reduce_mean(acts_real_pl, axis=0)
        mu_fake = tf.reduce_mean(acts_fake_pl, axis=0)

        sigma_real = tf_cov(acts_real_pl)
        sigma_fake = tf_cov(acts_fake_pl)
        diff = mu_real - mu_fake
        mu2 = tf.reduce_sum(tf.multiply(diff, diff))

        # Computing the sqrt of sigma_real * sigma_fake
        # See https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
        sqrt_sigma = tf_sqrtm(sigma_real)
        sqrt_a_sigma_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_fake, sqrt_sigma))

        tr = tf.trace(sigma_real + sigma_fake - 2 * tf_sqrtm(sqrt_a_sigma_a))
        fid = mu2 + tr

    def _fid(Xreal, Xfake, batch_size=16):
        acts_real_val = []
        print("Computing activations on real data")
        for X in progressbar.ProgressBar()(range(0, len(Xreal), batch_size)):
            if len(Xreal[X:X+batch_size]) == batch_size:
                acts_real_val.append(sess.run(acts_real, feed_dict={pl_real: Xreal[X:X+batch_size]}))

        acts_fake_val = []
        print("Computing activations on fake data")
        for X in progressbar.ProgressBar()(range(0, len(Xfake), batch_size)):
            if len(Xfake[X:X + batch_size]) == batch_size:
                acts_fake_val.append(sess.run(acts_fake, feed_dict={pl_fake: Xfake[X:X+batch_size]}))

        acts_fake_val = np.concatenate(acts_fake_val)
        acts_real_val = np.concatenate(acts_real_val)
        print("Computing composite FID")
        total_fid = 0
        for X in progressbar.ProgressBar()(range(0, len(acts_real_val), 100)):
            if len(acts_real_val[X:X + batch_size]) == batch_size:
                total_fid += sess.run(fid, feed_dict={acts_real_pl: acts_real_val[X:X+batch_size], acts_fake_pl: acts_fake_val[X:X+batch_size]})
        return total_fid
    return _fid

