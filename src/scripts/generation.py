import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from keras.models import load_model


def generate(generator,
             nz, zx, convdims,
             samples=1,
             filtering=True,
             threshold=True,
             tricatti=True):

    z_sample1 = np.random.uniform(-1., 1., (samples, nz) +
                                  (zx, ) * convdims)

    # Making the prediction
    model = generator.predict(z_sample1)

    model = (model + 1) * 0.5  # Convert from [-1,1] to [0,1]

    # A whole lot of complicated treatments
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

    return model


def main():
    generator = load_model(sys.argv[1])
    zx = 20
    convdims = 2
    nz = 1
    model = generate(generator, nz, zx, convdims, samples=10)

    plt.figure(figsize=(8, 8))
    plt.imshow(model[0, :, :], cmap='gray')

    h5_filename = "2D_Gen_" + '_' + str(zx)
    f = h5py.File(h5_filename + '.hdf5', mode='w')
    f.create_dataset('features', data=model)
    f.flush()
    f.close()


if __name__ == "__main__":
    main()
