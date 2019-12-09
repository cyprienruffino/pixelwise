import sys
import os
import h5py

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import copy

TRAIN = 0.85
VALID = 0.05
SYNTH = 0.05
VALIDSYNTH = 0.05

TEST = 0.9
TESTSYNTH = 0.1


def gen_const(batch, p):
    out = np.zeros(batch.shape)

    for img in range(len(batch)):
        n = int(np.size(batch[img]) * float(p))
        x = np.random.choice(np.arange(batch[img].shape[0]), n)
        y = np.random.choice(np.arange(batch[img].shape[1]), n)
        out[img, x, y] = batch[img, x, y]

    return out


def gen_celeba(filepath, const_ratio=0.05, ratio_xtain=.66, root = '../../celebrity_100K/'):
    
    xtrain = []
    xtest = []
    nbImg = len(glob.glob(root + '*.jpg'))* .75
    
    for idx, filename in enumerate(glob.glob(root + '*.jpg')):
        if idx > nbImg:
            break
        
        im=plt.imread(filename)
        
        if len(xtrain) <= ratio_xtain * nbImg:
            #xtrain.append(copy.deepcopy((im / 128) - 1))
            xtest.append(copy.deepcopy(im))
        else:
            #xtest.append(copy.deepcopy((im / 128) - 1))
            xtest.append(copy.deepcopy(im))
        print(idx)
        

    xtrain = np.array([(elem / 128) - 1 for elem in xtrain]) #(xtrain / 128) - 1
    xtest = np.array([(elem / 128) - 1 for elem in xtest]) #(xtest / 128) - 1

    xtrain = np.array(xtrain)
    xtest = np.array(xtest)
    
    np.random.shuffle(xtrain)
    np.random.shuffle(xtest)

    trainlen = int(TRAIN * len(xtrain))
    synthlen = int(SYNTH * len(xtrain))

    xreal = xtrain[0:trainlen]
    xsynth = xtrain[trainlen:trainlen + synthlen]

    validlen = int(VALID * len(xtrain))
    validsynthlen = int(VALIDSYNTH * len(xtrain))

    xvalid = xtrain[trainlen + synthlen:trainlen + synthlen + validlen]
    xvalidsynth = xtrain[trainlen + synthlen + validlen:trainlen + synthlen + validlen + validsynthlen]

    testlen = int(TEST * len(xtest))
    testsynthlen = int(TESTSYNTH * len(xtest))
    xtestreal = xtest[0:testlen]
    xtestsynth = xtest[testlen:testlen + testsynthlen]

    creal = gen_const(xreal, const_ratio)
    cgen = gen_const(xsynth, const_ratio)
    cvalid = gen_const(xvalidsynth, const_ratio)
    ctest = gen_const(xtestsynth, const_ratio)

    with h5py.File('./celeba.hdf5', "w") as f:
        f.create_dataset("xtrain", data=xreal)
        f.create_dataset("ctrain", data=creal)
        f.create_dataset("cgen", data=cgen)
        f.create_dataset("xvalid", data=xvalid)
        f.create_dataset("cvalid", data=cvalid)
        f.create_dataset("xtest", data=xtestreal)
        f.create_dataset("ctest", data=ctest)
        f.flush()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        gen_celeba(sys.argv[1])
    elif len(sys.argv) == 3:
        gen_celeba(sys.argv[1], sys.argv[2])
    else:
        print("Usage : gen_stable_cifar10.py output_path [constraints_ratio]")
