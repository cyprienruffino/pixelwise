import os
import sys

import h5py
import numpy as np
import pandas as pd
import progressbar
import tensorflow as tf
from tensorflow import keras as K

from metrics import create_fid_func, mse
from utils import noise


def get_best_epoch_weights(csvpath, runpath):
    df = pd.read_csv(csvpath)
    MSEs = np.array(df["MSE"])
    FIDs = np.array(df["FID"])

    normMSE = (MSEs - np.min(MSEs)) / (np.max(MSEs) - np.min(MSEs))
    normFID = (FIDs - np.min(FIDs)) / (np.max(FIDs) - np.min(FIDs))
    crit = np.sqrt(np.square(normMSE) + np.square(normFID))
    best_epoch = np.argmin(crit)

    return runpath + os.sep + "checkpoints" + os.sep + "G_" + str(best_epoch) + ".hdf5"


def eval_test(csvspath, runspath, fid, test_data, test_consts, out_path, batch_size=8):
        vals = {}
        z = noise.uniform(7)
        df = pd.DataFrame(columns=["Lambda", "FID", "MSE"])
        bar = progressbar.ProgressBar(maxvalue=len(os.listdir(csvspath)), redirect_stdout=True)
        for csv in bar(os.listdir(csvspath)):

            lmbda = csv.split("_")[-1].replace(".csv", "")
            if lmbda[-1] == ".":
                lmbda = lmbda[:-1]
            lmbda = float(lmbda)

            if lmbda not in vals:
                vals[lmbda] = [[], []]

            wpath = get_best_epoch_weights(csvspath + os.sep + csv, runspath + os.sep + csv.replace(".csv", ""))
            model = K.models.load_model(wpath)
            preds = []

            for i in range(0, len(test_consts), batch_size):
                preds += list(model.predict([z(batch_size), test_consts[i:i+batch_size]]))

            tfid = fid(test_data, preds, batch_size=batch_size)
            tmse = mse(preds, test_consts)
            vals[lmbda][0].append(tfid)
            vals[lmbda][1].append(tmse)

            df = df.append({"Lambda": float(lmbda), "FID": tfid, "MSE": tmse}, ignore_index=True)
        df.to_csv(out_path + os.sep + "values.csv")


def eval(csvspath, runs_path, model_path, dataset_path, output_path):

    with h5py.File(dataset_path, "r") as data:
        xtest = data["xtest"][:]
        ctest = data["ctest"][:]

    with tf.Session() as sess:
        fid = create_fid_func(model_path)
        eval_test(csvspath, runs_path, fid, xtest, ctest, output_path)


if __name__ == "__main__":
    if len(sys.argv) == 6:
        eval(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print("Usage : eval_test.py csvs_path runs_path fidmodel_path dataset_path output_path")
