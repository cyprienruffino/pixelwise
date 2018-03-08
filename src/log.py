import os
import h5py

import numpy as np

from PIL import Image


def write_config(config, file):
    with open(file, "w") as f:
        for prop, value in vars(config).items():
            f.write(prop + ": " + str(value) + "\n")


def create_dirs(logs_dir, checkpoints_dir, samples_dir):
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    if not os.path.exists(samples_dir):
        os.mkdir(samples_dir)


def save_jsons(D, G, DG, Adv, logs_dir):
    with open(logs_dir + "/D.json", "w") as f:
        f.write(D.to_json())
    with open(logs_dir + "/G.json", "w") as f:
        f.write(G.to_json())
    with open(logs_dir + "/DG.json", "w") as f:
        f.write(DG.to_json())
    with open(logs_dir + "/Adv.json", "w") as f:
        f.write(Adv.to_json())


def plot_models(D, G, DG, Adv, logs_dir):
    from keras.utils import plot_model

    plot_model(D, logs_dir + "/D.png")
    plot_model(G, logs_dir + "/G.png")
    plot_model(DG, logs_dir + "/DG.png")
    plot_model(Adv, logs_dir + "/Adv.png")


def plot_losses(D_losses, G_losses, logs_dir):
    import matplotlib.pyplot as plt

    plt.plot([i for i in range(1, 10)])
    plt.plot([i for i in range(10, 1, -1)])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    plt.savefig(logs_dir + "/losses.png")


def gen_hdf5(sample, samples_dir, run_name, epoch):
    f = h5py.File(samples_dir + run_name + "_" + str(epoch) + ".hdf5", mode="w")
    f.create_dataset('features', data=sample)
    f.flush()
    f.close()


def gen_png(sample, samples_dir, run_name, epoch):
    out = np.squeeze((sample + 1.) * 128.)
    image = Image.fromarray(np.uint8(out))
    image.save(samples_dir + run_name + "_" + str(epoch) + ".png")


def setup_tensorboard(logs_dir, run_name):
    from keras.backend import get_session
    import tensorflow as tf

    return tf.summary.FileWriter(logs_dir + "/" + run_name, get_session().graph)


def tensorboard_log_image(sample, writer, epoch):
    import tensorflow as tf
    # Logging samples
    data = np.transpose(sample, (0, 2, 3, 1))
    sample_pl = tf.placeholder(tf.float32, shape=data.shape, name='img')

    with tf.Session() as sess:
        samples_summary = sess.run(
            tf.summary.image(str(epoch), sample_pl),
            feed_dict={
                sample_pl: data
            })

    writer.add_summary(samples_summary, global_step=epoch)
    writer.flush()


def tensorboard_log_losses(D_loss, G_loss, writer, epoch):
    import tensorflow as tf

    # Logging losses
    losses_summary = tf.Summary(value=[
        tf.Summary.Value(tag="D_cost", simple_value=D_loss),
        tf.Summary.Value(tag="G_cost", simple_value=G_loss)
    ])

    writer.add_summary(losses_summary, global_step=epoch)
    writer.flush()


def save_summaries(D, G, DG, Adv, logs_dir):
    with open(logs_dir + "/Dsummary.txt", 'w') as fh:
        D.summary(print_fn=lambda x: fh.write(x + '\n'))

    with open(logs_dir + "/Gsummary.txt", 'w') as fh:
        G.summary(print_fn=lambda x: fh.write(x + '\n'))
