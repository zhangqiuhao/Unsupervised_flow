import sys, os
import numpy as np
from numpy.linalg import inv
from math import degrees
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

def main(args):
    np.set_printoptions(suppress=True)

    try:
        sequence = int(args[1])
    except:
        sequence = 9

    inpath_tf_oxt = '/mrtstorage/datasets/kitti/odometry/data_odometry_groundtruth/dataset/poses/08.txt'
    inpath_tf_is = '/home/zhang/odo.txt'

    with open(inpath_tf_oxt, 'r') as f:
        oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]
    with open(inpath_tf_is, 'r') as f:
        odo = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    points_oxt = [((l[2,3], -l[0,3])) for l in oxt]
    points_estimation1 = [((l[0,3], l[1,3])) for l in odo]

    ########
    start = 0
    end = 4000
    ########

    points_oxt = points_oxt[start:end]
    points_estimation1 = points_estimation1[1:end]

    path_oxt = Path(points_oxt, [Path.MOVETO] + [Path.LINETO] * (len(points_oxt) - 1))
    path_estimation1 = Path(points_estimation1, [Path.MOVETO] + [Path.LINETO] * (len(points_estimation1)-1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=True, c='#cccccc')
    ax.legend(handles=[patches.Patch(color='C0', label='OXTS Ground Truth'), patches.Patch(color='C1', label='Unflow')])
    ax.add_patch(patches.PathPatch(path_oxt, facecolor='none', edgecolor='C0'))
    ax.add_patch(patches.PathPatch(path_estimation1, facecolor='none', edgecolor='C1'))
    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('x in meter')
    ax.set_ylabel('y in meter')

    plt.autoscale(enable=True)
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
