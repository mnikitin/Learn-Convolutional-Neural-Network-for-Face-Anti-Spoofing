#!/usr/bin/env python

import sys, os, cv2, math, time
from glob import glob
import numpy as np

def compute_mean_std_casia(db_dir):
    mean_r, mean_g, mean_b = [], [], []
    std_r, std_g, std_b = [], [], []
    for frame_name in glob("%s/*/*/*.jpg" % (db_dir)):
        img = cv2.imread(frame_name)
        mean_r.append(np.mean(img[:,:,2]))
        mean_g.append(np.mean(img[:,:,1]))
        mean_b.append(np.mean(img[:,:,0]))
        std_r.append(np.std(img[:,:,2]))
        std_g.append(np.std(img[:,:,1]))
        std_b.append(np.std(img[:,:,0]))
    mean = [np.mean(mean_r), np.mean(mean_g), np.mean(mean_b)]
    std = [np.mean(std_r), np.mean(std_g), np.mean(std_b)]
    return mean, std


def main(argc, argv):
    db_dir = 'data/casia'
    scales = [1.0, 1.4, 1.8, 2.2, 2.6]
    for scale in scales:
        print("computing mean-std for scale: %s" % scale)
        cur_db_dir = db_dir + '/scale_' + str(scale) + '/train_release'
        mean, std = compute_mean_std_casia(cur_db_dir)
        with open(db_dir + '/mean_std__scale_' + str(scale) + '.txt', 'w') as f:
            f.write('%f %f %f\n' % (mean[0], mean[1], mean[2]))
            f.write('%f %f %f' % (std[0], std[1], std[2]))


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
