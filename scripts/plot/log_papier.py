#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

output_path = "./"
df = pd.read_csv("data/mnist_2/reeval/values0.csv")
"""
df = df.sort_values('Lambda')[116+44:]
len(df)
df.to_csv("data/mnist_2/reeval/values0.csv")
"""
df_lmb = pd.read_csv("data/noC/mnist_2/lmbda/values.csv")
df_nolmb = pd.read_csv("data/noC/mnist_2/nolmbda/values.csv")

ps = df["Lambda"].values
xs = df["FID"].values
ys = df["MSE"].values

nolmbda_fid = np.median(df_nolmb["FID"])
nolmbda_mse = np.median(df_nolmb["MSE"])
lmbda_fid = np.median(df_lmb["FID"])
lmbda_mse = np.median(df_lmb["MSE"])

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
plt.scatter(nolmbda_fid, nolmbda_mse, c="red", marker="^", label="λ=0")
plt.scatter(lmbda_fid, lmbda_mse, c="orange",  marker="s",label="No constraints")
plt.legend()
plt.savefig(output_path + os.sep + "pareto.pdf")
plt.gcf().clear()

mxs, mys, mps = sort_by_index((mxs, mys, mps), 2)

plt.gca().invert_xaxis()
plt.xscale('log')
plt.xlabel("lambda")
plt.ylabel("FID")
plt.plot(mps, mxs, 'b-', mps, mxs, 'bo')
plt.axhline(nolmbda_fid, color="red", linestyle="--",label="λ=0")
plt.axhline(lmbda_fid, color="green", label="No constraints")
plt.legend()
plt.gca().invert_xaxis()
plt.savefig(output_path + os.sep + "FID.pdf")
plt.gcf().clear()

plt.gca().invert_xaxis()
plt.xscale('log')
plt.xlabel("lambda")
plt.ylabel("MSE")
plt.plot(mps, mys, 'b-', mps, mys, 'bo')
plt.axhline(nolmbda_mse, color="red", linestyle="--", label="λ=0")
plt.axhline(lmbda_mse, color="green", label="No constraints")
plt.legend()
plt.gca().invert_xaxis()
plt.savefig(output_path + os.sep + "MSE.pdf")
