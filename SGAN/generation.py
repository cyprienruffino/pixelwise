import pickle
import time
import h5py

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import medfilt


def generate(generator,
             sgancfg,
             samples=1,
             filtering=True,
             threshold=True,
             tricatti=True):

    config = pickle.load(sgancfg)
    np.random.seed = config.seed

    from keras import load_model  # The seed must be set before importing Keras

    generator = load_model(generator)
    t_start = time.time()
    z_sample1 = np.random.uniform(-1., 1., (samples, config.nz, 10, 10))

    model = generator.predict(z_sample1)[:, 0, :, :]

    model = (model + 1) * 0.5  # Convert from [-1,1] to [0,1]

    if filtering:

        for ii in range(model.shape[0]):
            model[ii, :] = medfilt(model[ii, :], kernel_size=(3, 3))

    if threshold and not (tricatti):
        #        for ii in xrange(model.shape[0]):
        #            threshold=filters.threshold_otsu(model[ii,:])
        #            model[ii,:][model[ii,:]<threshold]=0
        #            model[ii,:][model[ii,:]>=threshold]=1
        threshold = 0.5
        model[model < threshold] = 0
        model[model >= threshold] = 1

    if threshold and tricatti:
        model[model < 0.334] = 0
        model[model >= 0.667] = 2
        model[np.where((model > 0) & (model < 2))] = 1
        model = model / 2.0

    print('elapsed_time is: ', time.time() - t_start)
    plt.figure(figsize=(8, 8))
    plt.imshow(model[0, :, :], cmap='gray')

    h5_filename = "2D_Gen_" + '_' + str(config.zx)
    f = h5py.File(h5_filename + '.hdf5', mode='w')
    f.create_dataset('features', data=model)
    f.flush()
    f.close()
