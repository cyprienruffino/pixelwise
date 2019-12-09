import os

import progressbar
import tensorflow as tf
import numpy as np
from tensorflow import keras as k
from tensorflow.keras import layers as kl


def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx


def tf_sqrtm(mat, eps=1e-10):
    # WARNING : This only works for symmetric matrices !
    s, u, v = tf.svd(mat)
    si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
    return tf.matmul(tf.matmul(u, tf.diag(si)), v, transpose_b=True)


def constraints_image(sample, constraints):
    if sample.shape[-1] == 1:
        image = np.stack((np.squeeze(sample, axis=-1),) * 3, -1)
        image = (((image + 1) / 2) * 255)

    else:
        image = (((sample + 1) / 2) * 255)

    for x in range(image.shape[1]):
        for y in range(image.shape[2]):
            if np.sum(constraints[0, x, y]) != 0:
                pix_sqerr = np.square(
                    np.sum(sample[0, x, y] - constraints[0, x, y]))
                if pix_sqerr > 0.1:
                    image[0, x, y] = [255, 0, 0]
                else:
                    image[0, x, y] = [0, 255, 0]

    return image


lmbda = 0.1
lr = 0.0002
batch_size = 64
epochs = 100

(x_train, _), (x_test) = k.datasets.mnist.load_data()

data = np.expand_dims((x_train / 128) - 1, axis=-1)
const = np.zeros((data.shape[0], 2))
const[:, 0] = np.mean(np.squeeze(data), axis=(1, 2))
const[:, 1] = np.std(np.squeeze(data), axis=(1, 2))

mu_means = np.mean(const[:, 0])
sigma_means = np.std(const[:, 0])

mu_stds = np.mean(const[:, 1])
sigma_stds = np.std(const[:, 1])

data_test = np.expand_dims((x_train / 128) - 1, axis=-1)
const_test = np.zeros((data_test.shape[0], 2))
const_test[:, 0] = np.mean(np.squeeze(data_test), axis=(1, 2))
const_test[:, 1] = np.std(np.squeeze(data_test), axis=(1, 2))

mu_means_test = np.mean(const_test[:, 0])
sigma_means_test = np.std(const_test[:, 0])

mu_stds_test = np.mean(const_test[:, 1])
sigma_stds_test = np.std(const_test[:, 1])


with tf.Session() as sess:

    d_in = tf.placeholder('float32', (None, 28, 28, 1))
    g_in = tf.placeholder('float32', (None, 100))
    c_in = tf.placeholder('float32', (None, 2))

    with tf.name_scope("Gen"):

        Z = kl.Input((100,), tensor=g_in, name="Z")
        C = kl.Input((2,), tensor=c_in, name="C")

        layer = kl.concatenate([Z, C])
        layer = kl.Dense(980, activation="relu")(layer)
        layer = kl.BatchNormalization()(layer)
        layer = kl.Reshape((7, 7, 20))(layer)

        for i in range(1, 3):
            layer = kl.Conv2DTranspose(
                filters=64 * (3 - i),
                kernel_size=3,
                padding="same",
                strides=2,
                activation="relu")(layer)
            layer = kl.BatchNormalization()(layer)

        G_out = kl.Conv2D(
            filters=1,
            kernel_size=3,
            activation="tanh",
            padding="same")(layer)

        G = k.Model(inputs=[Z, C], outputs=G_out)

    with tf.name_scope("Disc"):
        X = kl.Input((28, 28, 1), tensor=d_in, name="X")
        layer = kl.GaussianNoise(stddev=0.1)(X)

        # Discriminator
        for i in range(1, 3):
            layer = kl.Conv2D(
                filters=64 * i,
                kernel_size=3,
                padding="same",
                strides=2)(layer)
            layer = kl.LeakyReLU()(layer)
            layer = kl.BatchNormalization()(layer)

        layer = kl.Flatten()(layer)
        layer = kl.concatenate([layer, C], axis=-1)
        layer = kl.Dense(980)(layer)
        layer = kl.LeakyReLU()(layer)
        layer = kl.BatchNormalization()(layer)
        D_out = kl.Dense(1, activation="sigmoid")(layer)

        D = k.Model(inputs=[X, C], outputs=D_out)

    D_out = D.output
    G_out = G.output

    with tf.name_scope("DG"):
        DG = D([G_out, C])

    # Objectives
    with tf.name_scope("D_real_objective"):
        D_real_objective = tf.reduce_mean(-tf.log(D_out + 1e-8))

    with tf.name_scope("D_fake_objective"):
        D_fake_objective = tf.reduce_mean(-tf.log(1 - DG + 1e-8))

    with tf.name_scope("D_objective"):
        D_objective = D_real_objective + D_fake_objective

    with tf.name_scope("ML_cost"):
        mus = C[0]
        sigmas = C[1]
        gaus = tf.exp(-0.5 * tf.square((G_out - mus) / sigmas)) / \
            (tf.sqrt(2 * tf.constant(np.pi)) * sigmas)
        mle = tf.reduce_prod(gaus)

    with tf.name_scope("KL_cost"):
        mus_real = C[0]
        sigmas_real = C[1]
        mus_fake, sigmas_sq_fake = tf.nn.moments(G_out, axes=(1, 2))
        sigmas_fake = tf.sqrt(sigmas_sq_fake)

        kld = tf.reduce_sum(tf.log(sigmas_real/sigmas_fake)
                            + ((tf.square(sigmas_fake) + tf.square(mus_fake-mus_real))/2*tf.square(sigmas_real))
                            - 0.5)

    with tf.name_scope("G_objective"):
        G_real_objective = tf.reduce_mean(-tf.log(DG + 1e-8))
        G_objective = G_real_objective - (lmbda * kld)

    with tf.name_scope("FID"):
        cl = k.models.load_model("fid_models/mnist.hdf5")
        data_shape = cl.layers[0].input_shape
        pl_real = tf.placeholder(tf.float32, data_shape)
        pl_fake = tf.placeholder(tf.float32, data_shape)

        acts_real = k.Model(
            cl.input, cl.layers[-2].output)(kl.Input(tensor=pl_real))
        acts_fake = k.Model(
            cl.input, cl.layers[-2].output)(kl.Input(tensor=pl_fake))

        acts_real_pl = tf.placeholder(tf.float32, cl.layers[-2].output.shape)
        acts_fake_pl = tf.placeholder(tf.float32, cl.layers[-2].output.shape)

        mu_real = tf.reduce_mean(acts_real_pl, axis=0)
        mu_fake = tf.reduce_mean(acts_fake_pl, axis=0)
        sigma_real = tf_cov(acts_real_pl)
        sigma_fake = tf_cov(acts_fake_pl)
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

    mls = tf.summary.scalar("Likelihood", mle)
    klds = tf.summary.scalar("KL_divergence", kld)
    ds = tf.summary.scalar("D_cost", D_objective)
    gs = tf.summary.scalar("G_cost", G_real_objective)

    # Optimizers
    D_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    G_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    D_cost = D_optimizer.minimize(D_objective, var_list=D.trainable_weights)
    G_cost = G_optimizer.minimize(G_objective, var_list=G.trainable_weights)

    # Logging images
    imgpl = tf.placeholder(tf.float32, shape=(1, 28, 28, 1))
    trueimgpl = tf.placeholder(tf.float32, shape=(1, 28, 28, 1))

    imgsumm = tf.summary.merge([
        tf.summary.image("Generated", imgpl),
        tf.summary.image("Ground_truth", trueimgpl)
    ])

    writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
    if not os.path.exists("logs"):
        os.mkdir("logs/")
    writer.flush()

    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        bar = progressbar.ProgressBar(
            maxvalue=data.shape[0], redirect_stdout=True)
        print("Epoch" + str(epoch))

        for it in bar(range(0, data.shape[0], batch_size)):
            # Training D
            noise = np.random.uniform(-1, 1, (batch_size, 100,))
            batch = data[it:it + batch_size]
            consts = const[it:it + batch_size]

            if (batch.shape[0] < batch_size) or (consts.shape[0] < batch_size):
                continue

            _, dout = sess.run([D_cost, ds], feed_dict={X: batch, Z: noise, C: consts})

            # Training G
            noise = np.random.uniform(-1, 1, (batch_size, 100,))
            mu = np.random.normal(mu_means, sigma_means, (batch_size,))
            sigma = np.random.normal(mu_stds, sigma_stds, (batch_size,))

            _, gout, mout = sess.run([G_cost, gs, mls], feed_dict={Z: noise, C: np.array(list(zip(mu, sigma)))})

            t = (epoch * data.shape[0]) + it
            writer.add_summary(dout, t)
            writer.add_summary(gout, t)
            writer.add_summary(mout, t)

        # Logging images
        true_img = np.expand_dims(data[np.random.randint(len(data))], axis=0)
        c = np.expand_dims(const[np.random.randint(len(const))], axis=0)
        noise = np.random.uniform(-1, 1, (1, 100,))
        img = sess.run(G_out, feed_dict={Z: noise, C: c})

        imgout = sess.run(
            imgsumm, feed_dict={
                trueimgpl: true_img,
                imgpl: img
            })

        print("Validation")
        # Logging metrics
        generated = []
        generated_consts = []
        valid_data = []
        valid_consts = []
        bar = progressbar.ProgressBar(
            maxvalue=data_test.shape[0], redirect_stdout=True)
        for it in bar(range(0, data_test.shape[0], batch_size)):
            noise = np.random.uniform(-1, 1, (batch_size, 100,))
            batch = data_test[it:it + batch_size]
            consts = const_test[it:it + batch_size]
            if batch.shape[0] < batch_size:
                continue

            out = sess.run(G_out, feed_dict={Z: noise, C: consts})
            generated += list(out)
            generated_consts += list(np.concatenate((np.mean(out, axis=(1, 2)), np.std(out, axis=(1, 2))), axis=-1))
            valid_data += list(batch)
            valid_consts += list(consts)

        generated = np.asarray(generated)
        valid_data = np.asarray(valid_data)
        valid_consts = np.asarray(valid_consts)

        acts_real_val = []
        print("Computing activations on real data")
        for i in progressbar.ProgressBar()(range(0, len(data_test), batch_size)):
            if len(data_test[i:i + batch_size]) == batch_size:
                acts_real_val.append(sess.run(acts_real, feed_dict={
                                     pl_real: data_test[i:i + batch_size]}))

        acts_fake_val = []
        print("Computing activations on fake data")
        for i in progressbar.ProgressBar()(range(0, len(generated), batch_size)):
            if len(generated[i:i + batch_size]) == batch_size:
                acts_fake_val.append(sess.run(acts_fake, feed_dict={
                                     pl_fake: generated[i:i + batch_size]}))

        acts_fake_val = np.concatenate(acts_fake_val)
        acts_real_val = np.concatenate(acts_real_val)
        print("Computing composite FID")
        total_fid = 0
        for i in progressbar.ProgressBar()(range(0, len(acts_real_val), batch_size * 50)):
            if len(acts_real_val[i:i + batch_size]) == batch_size:
                total_fid += sess.run(fid, feed_dict={
                                      acts_real_pl: acts_real_val[i:i + batch_size], acts_fake_pl: acts_fake_val[i:i + batch_size]})

        # Writing all logs as tensorboard
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="FID", simple_value=total_fid)]), epoch)
        writer.add_summary(imgout, epoch)
        writer.flush()

        G.save(os.path.join("logs", "checkpoints", "G_" + str(epoch) + 
".hdf5"),
               include_optimizer=False)
        D.save(os.path.join("logs", "checkpoints", "D_" + str(epoch) + 
".hdf5"),
               include_optimizer=False)

    writer.close()
