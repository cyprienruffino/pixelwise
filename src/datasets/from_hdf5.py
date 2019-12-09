import h5py
import numpy as np


def get_data_provider(file, batch_size):
    with h5py.File(file, "r") as data:
        x = data["xtrain"][:]
        c = data["ctrain"][:]

    def _data_gen():
        while True:
            p = np.random.permutation(len(x))
            xtrain = x[p]
            ctrain = c[p]

            for i in range(0, len(xtrain), batch_size):
                xbatch = xtrain[i:i+batch_size]
                cbatch = ctrain[i:i + batch_size]
                if len(xbatch) == batch_size and len(cbatch) == batch_size:
                    yield xbatch, cbatch

    generator = _data_gen()
    return lambda: next(generator)


def get_genconst_provider(file, batch_size):
    with h5py.File(file, "r") as data:
        cgen = data["cgen"][:]

    def _const_gen():
        while True:
            np.random.shuffle(cgen)

            for i in range(0, len(cgen), batch_size):
                batch = cgen[i:i+batch_size]
                if len(batch) == batch_size:
                    yield batch

    generator = _const_gen()
    return lambda: next(generator)


def get_valid_provider(file, batch_size):
    with h5py.File(file, "r") as data:
        xvalid = data["xvalid"][:]

    def _data_gen():
        while True:
            for i in range(0, len(xvalid), batch_size):
                batch = xvalid[i:i+batch_size]
                if len(batch) == batch_size:
                    yield batch

    generator = _data_gen()
    return lambda: next(generator)


def get_validconst_provider(file, batch_size):
    with h5py.File(file, "r") as data:
        cvalid = data["cvalid"][:]

    def _const_gen():
        while True:
            for i in range(0, len(cvalid), batch_size):
                batch = cvalid[i:i+batch_size]
                if len(batch) == batch_size:
                    yield batch

    generator = _const_gen()
    return lambda: next(generator)


def get_test_provider(file, batch_size):
    with h5py.File(file, "r") as data:
        xtest = data["xtest"][:]

    def _data_gen():
        while True:
            for i in range(0, len(xtest), batch_size):
                batch = xtest[i:i+batch_size]
                if len(batch) == batch_size:
                    yield batch

    generator = _data_gen()
    return lambda: next(generator)


def get_testconst_provider(file, batch_size):
    with h5py.File(file, "r") as data:
        ctest = data["ctest"][:]

    def _const_gen():
        while True:
            for i in range(0, len(ctest), batch_size):
                batch = ctest[i:i+batch_size]
                if len(batch) == batch_size:
                    yield batch

    generator = _const_gen()
    return lambda _: next(generator)
