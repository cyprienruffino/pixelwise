import os
from functools import partial
from random import shuffle

import h5py
import numpy as np


def load_hdf5(path, dname):
    with h5py.File(path, "r") as f:
        data = f[dname][:]

    if len(data.shape) < 3:
        data = np.expand_dims(data, axis=-1)
    return data


def get_data_provider(path, batch_size):
    img = filter(lambda x: "img" in x, os.listdir(os.path.join(path, "train")))
    imgfiles = list(sorted(map(lambda x: os.path.join(path, "train", x), img)))

    cst = filter(lambda x: "cst" in x, os.listdir(os.path.join(path, "train")))
    cstfiles = list(sorted(map(lambda x: os.path.join(path, "train", x), cst)))

    def _data_gen():
        while True:
            p = list(zip(imgfiles, cstfiles))
            shuffle(p)
            imgs, csts = zip(*p)

            for i in range(0, len(imgs), batch_size):
                xbatch = imgs[i:i + batch_size]
                cbatch = csts[i:i + batch_size]
                if len(xbatch) == batch_size and len(cbatch) == batch_size:
                    yield np.array(list(map(partial(load_hdf5, dname="train"), xbatch))), \
                          np.array(list((map(partial(load_hdf5, dname="train"), cbatch))))

    generator = _data_gen()
    return lambda: next(generator)


def get_genconst_provider(path, batch_size):
    cst = filter(lambda x: "cst" in x, os.listdir(os.path.join(path, "cgen")))
    cstfiles = list(sorted(map(lambda x: os.path.join(path, "cgen", x), cst)))

    def _data_gen():
        while True:
            shuffle(cstfiles)
            for i in range(0, len(cstfiles), batch_size):
                cbatch = cstfiles[i:i + batch_size]
                if len(cbatch) == batch_size:
                    yield np.array(list(map(partial(load_hdf5, dname="cgen"), cbatch)))

    generator = _data_gen()
    return lambda: next(generator)


def get_valid_provider(path, batch_size):
    img = filter(lambda x: "img" in x, os.listdir(os.path.join(path, "valid")))
    imgfiles = list(sorted(map(lambda x: os.path.join(path, "valid", x), img)))

    def _data_gen():
        while True:
            shuffle(imgfiles)
            for i in range(0, len(imgfiles), batch_size):
                xbatch = imgfiles[i:i + batch_size]
                if len(xbatch) == batch_size:
                    yield np.array(list(map(partial(load_hdf5, dname="valid"), xbatch)))

    generator = _data_gen()
    return lambda: next(generator)


def get_validconst_provider(path, batch_size):
    cst = filter(lambda x: "cst" in x, os.listdir(os.path.join(path, "cvalid")))
    cstfiles = list(sorted(map(lambda x: os.path.join(path, "cvalid", x), cst)))

    def _data_gen():
        while True:
            shuffle(cstfiles)
            for i in range(0, len(cstfiles), batch_size):
                cbatch = cstfiles[i:i + batch_size]
                if len(cbatch) == batch_size:
                    yield np.array(list(map(partial(load_hdf5, dname="cvalid"), cbatch)))

    generator = _data_gen()
    return lambda: next(generator)


def get_test_provider(path, batch_size):
    img = filter(lambda x: "img" in x, os.listdir(os.path.join(path, "test")))
    imgfiles = list(sorted(map(lambda x: os.path.join(path, "test", x), img)))

    def _data_gen():
        while True:
            shuffle(imgfiles)
            for i in range(0, len(imgfiles), batch_size):
                xbatch = imgfiles[i:i + batch_size]
                if len(xbatch) == batch_size:
                    yield np.array(list(map(partial(load_hdf5, dname="test"), xbatch)))

    generator = _data_gen()
    return lambda: next(generator)


def get_testconst_provider(path, batch_size):
    cst = filter(lambda x: ".hdf5" in x, os.listdir(os.path.join(path, "ctest")))
    cstfiles = list(sorted(map(lambda x: os.path.join(path, "ctest", x), cst)))

    def _data_gen():
        while True:
            shuffle(cstfiles)
            for i in range(0, len(cstfiles), batch_size):
                cbatch = cstfiles[i:i + batch_size]
                if len(cbatch) == batch_size:
                    yield np.array(list(map(partial(load_hdf5, dname="ctest"), cbatch)))

    generator = _data_gen()
    return lambda: next(generator)
