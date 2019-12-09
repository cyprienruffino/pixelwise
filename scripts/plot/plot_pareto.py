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


def get_vals_dict(csvspath, tag1, tag2):
    vals = {}
    for csv in os.listdir(csvspath):
        lmbda = float(csv.split("_")[-1].replace(".csv", ""))

        if lmbda not in vals:
            vals[lmbda] = [[], []]

        df = pd.read_csv(csvspath + os.sep + csv)
        vals[lmbda][0].append(np.array(df[tag1]))
        vals[lmbda][1].append(np.array(df[tag2].tolist()))

    return vals


def get_medians(vals):
    xs = []
    ys = []
    ps = []
    mxs = []
    mys = []
    mps = []

    for lmbda in vals.keys():
        vals1 = list(np.median(np.array(vals[lmbda][0]), axis=0))
        vals2 = list(np.median(np.array(vals[lmbda][1]), axis=0))
        xs += vals1
        ys += vals2
        ps += [float(lmbda)] * len(vals1)

        mxs.append(np.min(vals1))
        mys.append(vals2[np.argmin(vals1)])
        mps.append(float(lmbda))

    return (xs, ys, ps), (mxs, mys, mps)


def sort_by_index(tup, index):
    z = list(zip(*tup))
    z.sort(key=lambda x: x[index])
    return zip(*z)


def pareto(csvspath, tag1, tag2, output_path):

    vals = get_vals_dict(csvspath, tag1, tag2)
    (xs, ys, ps), (mxs, mys, mps) = get_medians(vals)

    xs, ys, ps = sort_by_index((xs, ys, ps), 0)
    par_x, par_y = pareto_frontier(xs, ys)

    plt.xlabel(tag1)
    plt.ylabel(tag2)
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
    plt.ylabel(tag1)
    plt.plot(mps, mxs, 'b-', mps, mxs, 'bo')
    plt.savefig(output_path + os.sep + tag1+".png")
    plt.gcf().clear()

    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel(tag2)
    plt.plot(mps, mys, 'b-', mps, mys, 'bo')
    plt.savefig(output_path + os.sep + tag2+".png")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        pareto(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Usage : pareto.py csvs_path scalar_1 scalar_2 output_path")
