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


def pareto(csvspath, tag1, tag2, output_file):

    xs = []
    ys = []
    ps = []
    for csv in os.listdir(csvspath):
        df = pd.read_csv(csvspath + os.sep + csv)

        xs += df[tag1].tolist()
        ys += df[tag2].tolist()
        for i in range(len(df[tag1].tolist())):
            ps.append(float(csv.replace(".csv", "")))

    z = list(zip(xs, ys, ps))
    z.sort(key=lambda x: x[0])
    xs, ys, ps = zip(*z)

    par_x, par_y = pareto_frontier(xs, ys)

    plt.xlabel(tag1)
    plt.ylabel(tag2)
    plt.scatter(xs, ys, c=ps, norm=matplotlib.colors.LogNorm())
    plt.plot(par_x, par_y, 'r-')
    bar = plt.colorbar()
    bar.set_label("lambda")

    plt.savefig(output_file)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        pareto(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Usage : pareto.py csvs_path scalar_1 scalar_2 output_file")
