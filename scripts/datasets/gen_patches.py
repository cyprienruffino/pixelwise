import os
import sys
from functools import partial

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf


TRAIN = 20000
SYNTH = 5000

VALID = 2000
VALIDSYNTH = 500

TEST = 4000
TESTSYNTH = 1000


def gen_const(imTex, const_ratio):
    out = np.zeros(imTex.shape)

    n = int(imTex.shape[0] * imTex.shape[1] * const_ratio)
    x = np.random.choice(np.arange(imTex.shape[0]), n)
    y = np.random.choice(np.arange(imTex.shape[1]), n)
    out[x, y] = imTex[x, y]

    return out


def cut_patch(imTex, npx):
    if npx < imTex.shape[0] and npx < imTex.shape[1]:  # sample patches
        h = np.random.randint(imTex.shape[0] - npx)
        w = np.random.randint(imTex.shape[1] - npx)
        img = imTex[h:h + npx, w:w + npx]
    else:  # whole input texture
        raise Exception('Patches are larger than the image')

    return img


def cut(i, imTex, npx, path, name=None, save_patch=True, save_const=True, const_ratio=0.001):
    patch = cut_patch(imTex, npx)
    if save_patch:
        with h5py.File(os.path.join(path, name, str(i) + 'img.hdf5'), 'w') as f:
            f.create_dataset(name, data=patch)

    const = gen_const(patch, const_ratio)
    if save_const:
        with h5py.File(os.path.join(path, name, str(i)+'cst.hdf5'), 'w') as f:
            f.create_dataset(name, data=const)


def build_sub_dataset(name, imTex, num_patches, npx, path, save_patch=True, save_const=True, const_ratio=0.001):

    for i in tqdm(range(num_patches)):
        cut(i, path=path, name=name, imTex=imTex, npx=npx, save_patch=save_patch, save_const=save_const, const_ratio=const_ratio)

    sys.stdout.flush()

"""
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def cut(i, imTex, npx, writer, name=None, save_patch=True, save_const=True):
    patch = cut_patch(imTex, npx)
    if save_patch:
        feature = {name+'/image': bytes_feature(tf.compat.as_bytes(patch.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    const = gen_const(patch)
    if save_const:
        feature = {name+'/const': bytes_feature(tf.compat.as_bytes(const.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


def build_sub_dataset(name, imTex, num_patches, npx, path, save_patch=True, save_const=True):
    writer = tf.python_io.TFRecordWriter(os.path.join(path, name+".tfrecord"))

    for i in tqdm(range(num_patches)):
        cut(i, name=name, imTex=imTex, npx=npx, writer=writer)

    writer.close()
    sys.stdout.flush()
"""


def build_dataset(image, npx, const_ratio, path):
    img = Image.open(image)
    imTex = (np.asarray(img) / 128) - 1

    os.mkdir(os.path.join(path, 'train'))
    os.mkdir(os.path.join(path, 'valid'))
    os.mkdir(os.path.join(path, 'test'))
    os.mkdir(os.path.join(path, 'cgen'))
    os.mkdir(os.path.join(path, 'cvalid'))
    os.mkdir(os.path.join(path, 'ctest'))

    print("Cutting train")
    build_sub_dataset("train", imTex, TRAIN, npx, path, const_ratio=const_ratio)

    print("Cutting valid")
    build_sub_dataset("valid", imTex, VALID, npx, path, save_const=False, const_ratio=const_ratio)

    print("Cutting test")
    build_sub_dataset("test", imTex, TEST, npx, path, save_const=False, const_ratio=const_ratio)

    print("Generating train constraints")
    build_sub_dataset("cgen", imTex, SYNTH, npx, path, save_patch=False, const_ratio=const_ratio)

    print("Generating valid constraints")
    build_sub_dataset("cvalid", imTex, VALIDSYNTH, npx, path, save_patch=False, const_ratio=const_ratio)

    print("Generating test constraints")
    build_sub_dataset("ctest", imTex, TESTSYNTH, npx, path, save_patch=False, const_ratio=const_ratio)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python gen_patches.py image_file npx const_ratio output")
    else:
        build_dataset(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
