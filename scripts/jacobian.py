import h5py
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
#from gradients import jacobian as jc
from tensorflow.python.ops.parallel_for import jacobian


def jaco(generator_path, noise, output_path):
    generator = load_model(generator_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        jac = sess.run(jacobian(generator.output, generator.input),
                       feed_dict={generator.input: noise})
        print(jac)

    f = h5py.File(output_path, mode='w')
    f.create_dataset('features', data=jac)
    f.flush()
    f.close()


def main():
    jaco(sys.argv[1], np.random.uniform(-1, 1, (1, 2)), 'jacobian.hdf5')


if __name__ == "__main__":
    main()
