import h5py
import sys
import numpy as np
from keras.layers import Lambda
from keras.models import load_model, Model
import keras.backend as K


def create_jacobian_evaluator(generator):
    inp = generator.input
    grads = Lambda(lambda x: K.gradients(generator(x), x))(inp)
    return Model(inp, grads)


def main():
    generator = load_model(sys.argv[1])
    zx = 20
    convdims = 2
    nz = 1

    jacobian_evaluator = create_jacobian_evaluator(generator)

    z_sample = np.random.uniform(-1., 1., (1, nz) + (zx, ) * convdims)

    jacobian = jacobian_evaluator.predict(z_sample)

    print(jacobian)

    h5_filename = "Jacobian_" + '_' + str(zx)
    f = h5py.File(h5_filename + '.hdf5', mode='w')
    f.create_dataset('features', data=jacobian)
    f.flush()
    f.close()


if __name__ == "__main__":
    main()
