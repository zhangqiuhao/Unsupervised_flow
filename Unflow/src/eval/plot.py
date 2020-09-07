import sys, os
import numpy as np
from numpy.linalg import inv
from math import degrees
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


def KittiPlotTrajectories(sequence, num, odo_save_path, matrix_input=None):
    np.set_printoptions(suppress=True)

    ########
    start = 0
    end = num
    ########

    inpath_tf_oxt = '/home/zhang/pcl_data/gridmap_train/odo_gt/poses/'+sequence+'.txt'
    inpath_tf_is = odo_save_path + "/" + sequence + '_kitti.txt'

    if matrix_input is not None:
        with open(inpath_tf_oxt, 'r') as f:
            oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]
            points_oxt = [((l[0, 3], l[2, 3])) for l in oxt]
            points_oxt = points_oxt[start:end]
            path_oxt = Path(points_oxt, [Path.MOVETO] + [Path.LINETO] * (len(points_oxt) - 1))

    with open(inpath_tf_is, 'r') as f:
        odo = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]
        points_estimation1 = [((l[0, 3], l[2, 3])) for l in odo]
        points_estimation1 = points_estimation1[0:end]
        path_estimation1 = Path(points_estimation1, [Path.MOVETO] + [Path.LINETO] * (len(points_estimation1)-1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=True, c='#cccccc')
    ax.legend(handles=[patches.Patch(color='C0', label='OXTS Ground Truth'), patches.Patch(color='C1', label='Unflow')])
    if matrix_input is not None:
        ax.add_patch(patches.PathPatch(path_oxt, facecolor='none', edgecolor='C0'))
    ax.add_patch(patches.PathPatch(path_estimation1, facecolor='none', edgecolor='C1'))
    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('x in meter')
    ax.set_ylabel('z in meter')

    for node in range(0, num, int(num / 10)):
        if matrix_input is not None:
            ax.text(points_oxt[node][0], points_oxt[node][1], str(node), fontsize=8)
            ax.scatter(points_oxt[node][0], points_oxt[node][1], s=20)
        ax.text(points_estimation1[node][0], points_estimation1[node][1], str(node), fontsize=8)
        ax.scatter(points_estimation1[node][0], points_estimation1[node][1], s=20)

    plt.autoscale(enable=True)
    plt.savefig(odo_save_path + "/" + sequence + '_odometry.png')
