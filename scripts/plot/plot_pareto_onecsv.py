#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

import numpy as np
import pandas as pd


def pareto_frontier(Xs, Ys, maxX=False, maxY=False):

    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)

    p_front = [myList[0]]

    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)

    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY


def sort_by_index(tup, index):
    z = list(zip(*tup))
    z.sort(key=lambda x: x[index])
    return zip(*z)


def pareto(csvpath, output_path):

    df = pd.read_csv(csvpath)
    ps = df["Lambda"].values
    xs = df["FID"].values
    ys = df["MSE"].values

    vals = {}
    for ind, row in df.iterrows():
        if row["Lambda"] not in vals:
            vals[row["Lambda"]] = [[], []]
        vals[row["Lambda"]][0].append(row["FID"])
        vals[row["Lambda"]][1].append(row["MSE"])

    mxs = []
    mys = []
    mps = []
    for lmbda in vals.keys():
        mxs.append(np.median(vals[lmbda][0]))
        mys.append(np.median(vals[lmbda][1]))
        mps.append(lmbda)

    xs, ys, ps = sort_by_index((xs, ys, ps), 0)
    par_x, par_y = pareto_frontier(xs, ys)

    plt.xlabel("FID")
    plt.ylabel("MSE")
    plt.scatter(xs, ys, c=ps, norm=matplotlib.colors.LogNorm())
    plt.plot(par_x, par_y, 'r-')
    bar = plt.colorbar()
    bar.set_label("lambda")
    plt.savefig(output_path + os.sep + "pareto.png")
    plt.gcf().clear()

    mxs, mys, mps = sort_by_index((mxs, mys, mps), 2)

    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("FID")
    plt.plot(mps, mxs, 'b-', mps, mxs, 'bo')
    plt.savefig(output_path + os.sep + "FID.png")
    plt.gcf().clear()

    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.plot(mps, mys, 'b-', mps, mys, 'bo')
    plt.savefig(output_path + os.sep + "MSE.png")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        pareto(sys.argv[1], sys.argv[2])
    else:
        print("Usage : pareto.py csvs_path output_path")
