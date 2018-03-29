#!/usr/bin/env python

import math
import numpy as np

def eval_stat(scores, labels, thr):
    pred = scores >= thr
    TN = np.sum((labels == -1) & (pred == False))
    FN = np.sum((labels == 1) & (pred == False))
    FP = np.sum((labels == -1) & (pred == True))
    TP = np.sum((labels == 1) & (pred == True))
    return TN, FN, FP, TP

def get_thresholds(scores, grid_density):
    # uniform thresholds in [min, max]
    Min, Max = min(scores), max(scores)
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(Min + i * (Max - Min) / float(grid_density))
    return thresholds


def get_eer_stats(scores, labels, grid_density = 10000):
    thresholds = get_thresholds(scores, grid_density)
    min_dist = 1.0
    min_dist_stats = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_stat(scores, labels, thr)
        far = FP / float(TN + FP)
        frr = FN / float(TP + FN)
        dist = math.fabs(far - frr)
        if dist < min_dist:
            min_dist = dist
            min_dist_stats = [far, frr, thr]
    eer = (min_dist_stats[0] + min_dist_stats[1]) / 2.0
    thr = min_dist_stats[2]
    return eer, thr


def get_hter_at_thr(scores, labels, thr):
    TN, FN, FP, TP = eval_stat(scores, labels, thr)
    far = FP / float(TN + FP)
    frr = FN / float(TP + FN)
    hter = (far + frr) / 2.0
    return hter


def get_accuracy(scores, labels, thr):
    TN, FN, FP, TP = eval_stat(scores, labels, thr)
    accuracy = float(TP + TN) / len(scores)
    return accuracy

def get_best_thr(scores, labels, grid_density = 10000):
    thresholds = get_thresholds(scores, grid_density)
    acc_best = 0.0
    for thr in thresholds:
        acc = get_accuracy(scores, labels, thr)
        if acc > acc_best:
            acc_best = acc
            thr_best = thr
    return thr_best, acc_best
