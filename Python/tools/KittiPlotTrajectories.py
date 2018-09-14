import sys, os
import numpy as np
from numpy.linalg import inv
import pcl
from numpngw import write_png
from math import degrees
import matplotlib.pyplot as plt
from PlotDirectionalField import directionalField
import svgwrite
from matplotlib.path import Path
import matplotlib.patches as patches
from NumpyHelpers import XYAngleToMatrix2

def main(args):
    np.set_printoptions(suppress=True)

    try:
        sequence = int(args[1])
    except:
        sequence = 9

    inpath_tf_oxt = '/mrtstorage/datasets/kitti/odometry/all_py/poses/' + str(sequence).zfill(2) + '.txt'
    inpath_tf_estimation1 = '/home/klein/U/inference/cylindrical_ref_original_seq_9/estimation_kitti_poses.txt'
    inpath_tf_estimation2 = '/home/klein/U/inference/flownet_simple_seq_9/estimation_kitti_poses.txt'

    with open(inpath_tf_oxt, 'r') as f:
        oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    with open(inpath_tf_estimation1, 'r') as f:
        est1 = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    with open(inpath_tf_estimation2, 'r') as f:
        est2 = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    points_oxt = [((l[2,3], -l[0,3])) for l in oxt]
    points_estimation1 = [((l[2,3], -l[0,3])) for l in est1]
    points_estimation2 = [((l[2,3], -l[0,3])) for l in est2]

    ########
    start = 500
    end = 800
    ########

    points_oxt = points_oxt[start:end]
    points_estimation1 = points_estimation1[1:end]
    points_estimation2 = points_estimation2[1:end]

    path_oxt = Path(points_oxt, [Path.MOVETO] + [Path.LINETO] * (len(points_oxt) - 1))
    path_estimation1 = Path(points_estimation1, [Path.MOVETO] + [Path.LINETO] * (len(points_estimation1)-1))
    path_estimation2 = Path(points_estimation2, [Path.MOVETO] + [Path.LINETO] * (len(points_estimation2)-1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=True, c='#cccccc')
    ax.legend(handles=[patches.Patch(color='C0', label='OXT Ground Truth'), patches.Patch(color='C1', label='Own CNN'), patches.Patch(color='C2', label='FlowNet')])
    ax.add_patch(patches.PathPatch(path_oxt, facecolor='none', edgecolor='C0'))
    ax.add_patch(patches.PathPatch(path_estimation1, facecolor='none', edgecolor='C1'))
    ax.add_patch(patches.PathPatch(path_estimation2, facecolor='none', edgecolor='C2'))
    ax.set_aspect('equal', 'datalim')
    plt.autoscale(enable=True)

    plt.show()

if __name__ == "__main__":
    main(sys.argv)
