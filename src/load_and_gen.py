import sys
import os
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import progressbar

from layers.instancenorm import InstanceNormalization
from utils.log import constraints_image

def main():
    modpath = sys.argv[1]
    constpath = sys.argv[2]

    mod = tf.keras.models.load_model(modpath, custom_objects={"InstanceNormalization":InstanceNormalization})
    const = h5py.File(constpath)["train"][:]
    if len(const.shape) < 3:
        const = np.expand_dims(const, axis=-1)

    for i in progressbar.ProgressBar()(range(100)): 
        out = mod.predict([np.random.uniform(-1, 1, (1, 40, 40, const.shape[-1])), np.expand_dims(const, axis=0)]) 
        plt.imsave("figs/"+str(i)+".png",  constraints_image(out, np.expand_dims((const/2), axis=0))[0].astype("uint16")) 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python load_and_gen.py modpath constpath")
    else:
        main()