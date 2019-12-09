import os
import sys
import gc

import numpy as np
from multiprocessing import Pool

import tqdm
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl

from PIL import Image


HEIGHT = 160
WIDTH = 160


def load_img(path):
    img = Image.open(path)
    label = path.split('/')[-2]
    return np.array(img.resize((HEIGHT, WIDTH), Image.ANTIALIAS), dtype="float32"), label


def load_dtd(dpath):
    train = []
    val = []

    print("Loading the dataset")
    with open(os.path.join(dpath, 'labels', 'train1.txt')) as f:
        for file in f.readlines():
            train.append(os.path.join(dpath, "images/", file.replace('\n', '')))

    with open(os.path.join(dpath, 'labels', 'train1.txt')) as f:
        for file in f.readlines():
            val.append(os.path.join(dpath, "images/", file.replace('\n', '')))

    train += val[0:int(len(val)*0.8)]
    val = val[int(len(val)*0.8):]
            
    pool = Pool(8)
    labelsdict = {}
    labelcount = 0

    train_img_raw = []
    train_labels = []
    print("Loading train")
    for img, label in tqdm.tqdm(pool.imap_unordered(load_img, train), total=len(train)):
        train_img_raw.append(img)
        if label not in labelsdict:
            labelsdict[label] = labelcount
            labelcount += 1
        train_labels.append(labelsdict[label])

    val_img_raw = []
    val_labels = []
    print("Loading val")
    for img, label in tqdm.tqdm(pool.imap_unordered(load_img, val), total=len(val)):
        val_img_raw.append(img)
        if label not in labelsdict:
            labelsdict[label] = labelcount
            labelcount += 1
        val_labels.append(labelsdict[label])

    print("Preprocessing the data")
    prep = k.applications.inception_resnet_v2.preprocess_input
    print("Preprocessing train")
    train_img = []
    for img in tqdm.tqdm(pool.imap_unordered(prep, train_img_raw), total=len(train_img_raw)):
        train_img.append(img)
    del train_img_raw
    tri = np.asarray(train_img)
    del train_img

    print("Preprocessing val")
    val_img = []
    for img in tqdm.tqdm(pool.imap_unordered(prep, val_img_raw), total=len(val_img_raw)):
        val_img.append(img)
    del val_img_raw[:]
    vi = np.asarray(val_img)
    del val_img[:]

    gc.collect()
    trl = k.utils.to_categorical(train_labels)
    vl = k.utils.to_categorical(val_labels)

    return (tri, trl), (vi, vl)


def load_dtd_test(dpath):
    test = []
    with open(os.path.join(dpath, 'labels', 'train1.txt')) as f:
        for file in f.readlines():
            test.append(os.path.join(dpath, "images/", file.replace('\n', '')))

    pool = Pool(8)
    labelsdict = {}
    labelcount = 0
    test_img_raw = []
    test_labels = []
    print("Loading test")
    for img, label in tqdm.tqdm(pool.imap_unordered(load_img, test), total=len(test)):
        test_img_raw.append(img)
        if label not in labelsdict:
            labelsdict[label] = labelcount
            labelcount += 1
        test_labels.append(labelsdict[label])

    print("Preprocessing test")
    prep = k.applications.inception_resnet_v2.preprocess_input
    test_img = []
    for img in tqdm.tqdm(pool.imap_unordered(prep, test_img_raw), total=len(test_img_raw)):
        test_img.append(img)
    del test_img_raw[:]
    tei = np.asarray(test_img)
    del test_img[:]

    tel = k.utils.to_categorical(test_labels)

    return (tei, tel)


def create_classifier():
    print("Building model")
    inception = k.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    layer = kl.Flatten()(inception.output)
    layer = kl.Dense(1000, activation="relu")(layer)
    fidmodel = k.Model(inception.input, layer)
    layer = kl.Dropout(0.2)(layer)
    D_out = kl.Dense(47, activation="softmax")(layer)

    model = k.models.Model(inception.input, D_out)
    return model, fidmodel


def train_classifier(data_path):
    (x_train, y_train), (x_val, y_val) = load_dtd(data_path)

    model, fidmodel = create_classifier()
    model.compile(k.optimizers.Adam(0.00005), "categorical_crossentropy", ["accuracy"])

    callbacks = [
        k.callbacks.TensorBoard("./runs/fid/"),
        k.callbacks.ModelCheckpoint("fidmodel-{val_loss:.2f}.hdf5", monitor='val_loss', save_best_only=True, period=1)
        
    ]

    print("Training")
    model.fit(x_train, y_train, batch_size=2, epochs=30,
              validation_data=(x_val, y_val), callbacks=callbacks)
    del x_train
    del y_train
    del x_val
    del y_val
    gc.collect()
    #(x_test, y_test) = load_dtd_test(data_path)
    #model.evaluate(x_test, y_test)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        train_classifier(sys.argv[1])
    else:
        print("Usage: python train_dtd_fid.py dataset_path")
