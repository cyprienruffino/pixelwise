import tensorflow as tf

import numpy as np
import h5py
import os
import numpy
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.misc import imsave

from layers.instancenorm import InstanceNormalization


with h5py.File("2.hdf5", "r") as f:
    const = f["ctest"][:]
const.max()
const = (np.expand_dims(const, axis=0) / 255) 
const.mean()

noise1 = np.random.uniform(-1, 1, (1, 40, 40, 3))
noise2 = np.random.uniform(-1, 1, (1, 40, 40, 3))

model = load_model("./G_20.hdf5", custom_objects={"InstanceNormalization": InstanceNormalization})

data1 = model.predict([noise1, const])
data2 = model.predict([noise2, const])

imsave("1.png",np.squeeze((data1 +1) * 128))
imsave("2.png",np.squeeze((data2 +1) * 128))
plt.clf()
diff = np.sum(np.abs(np.squeeze((data1 +1) * 128) - np.squeeze((data2 +1) * 128)), axis=-1) / 3
plt.imshow(diff, cmap="Greys", vmin=0, vmax=255)
plt.colorbar()
plt.savefig("diff.png")


model = load_model("./G_30_pac.hdf5", custom_objects={"InstanceNormalization": InstanceNormalization})

data1 = model.predict([noise1, const])
data2 = model.predict([noise2, const])

imsave("1_pac.png",np.squeeze((data1 +1) * 128))
imsave("2_pac.png",np.squeeze((data2 +1) * 128))
plt.clf()
diff = np.sum(np.abs(np.squeeze((data1 +1) * 128) - np.squeeze((data2 +1) * 128)), axis=-1) / 3
plt.imshow(diff, cmap="Greys", vmin=0, vmax=255)
plt.colorbar()
plt.savefig("diff_pac.png")
