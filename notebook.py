#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import tensorflow.keras as K
from scipy.misc import imread, imshow, imsave
import numpy as np
from PIL import Image
import h5py
%matplotlib inline
import matplotlib.pyplot as plt

from src.datasets import from_files
from src.datasets import ti_patch
os.chdir("src")
from applications import edskips
os.chdir("..")
import pandas as pd
from layers import InstanceNormalization


mod = tf.keras.models.load_model("runs/channels_unet/G_22.hdf5", compile=False, custom_objects={"InstanceNormalization": InstanceNormalization})

path = "datasets/ti_2D_160/"
cst = filter(lambda x: "hdf5" in x, os.listdir(os.path.join(path, "cgen")))
cstfiles = list(sorted(map(lambda x: os.path.join(path, "cgen", x), cst)))


masks = []
imgs = []
consts = []
for i in range(100):
    with h5py.File(cstfiles[i], "r") as f:
        data = (f["cgen"][:] / 128) - 1

    ndata = np.expand_dims(np.expand_dims(data, axis=0), axis=-1)
    stacked = np.vstack([ndata]*20)
    # stacked = np.zeros((20, 160, 160, 1))

    noise = np.random.uniform(-1, 1, (20 ,40, 40, 1))
    # noise = np.zeros((20 ,40, 40, 1))

    imgs.append(np.squeeze(mod.predict([noise, stacked])))
    masks.append(np.squeeze(np.ceil(np.abs(stacked))))
    consts.append(np.squeeze(stacked))

imgs = np.stack(imgs, axis=0)
masks = np.stack(masks, axis=0)
consts = np.stack(consts, axis=0)
masks.max()
errs = (mask * out - consts)
plt.hist(errs[np.where(errs != 0)])
plt.imshow(out[12])
plt.imsave("tmp.png", out[12])
plt.imshow(out[14])
plt.imshow(out[12] - out[14])
