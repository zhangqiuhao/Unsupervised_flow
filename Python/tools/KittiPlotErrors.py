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
from NumpyHelpers import XYAngleToMatrix2, rotationMatrixToEulerAngles
import matplotlib.patches as mpatches


def main(args):
    np.set_printoptions(suppress=True)

    try:
        sequence = int(args[1])
    except:
        sequence = 9

    inpath_tf_oxt = '/mrtstorage/datasets/kitti/odometry/all_py/poses/' + str(sequence).zfill(2) + '.txt'
    inpath_tf_estimation1 = '/home/klein/U/inference/cylindrical_ref_original_seq_9/estimation_kitti_poses.txt'
    inpath_tf_estimation2 = '/home/klein/U/inference/flownet_simple_seq_9/estimation_kitti_poses.txt'

    inpath_calib = '/mrtstorage/datasets/kitti/odometry/data_odometry_calib/sequences/' + str(sequence).zfill(2) + '/calib.txt'
    with open(inpath_calib, 'r') as f:
        data_calib = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(': ')[1].split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    tf_cam_to_velo = data_calib[4]
    tf_velo_to_cam = inv(data_calib[4])

    with open(inpath_tf_oxt, 'r') as f:
        oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    with open(inpath_tf_estimation1, 'r') as f:
        est1 = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    with open(inpath_tf_estimation2, 'r') as f:
        est2 = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]


    ########
    start = 500
    end = 800
    ########


    angles1 = []
    angles2 = []
    angles_gt = []
    dx1 = []
    dx2 = []
    dx_gt = []
    dy1 = []
    dy2 = []
    dy_gt = []

    for i, tf_this_frame in enumerate(oxt[:end-1]):
        if i < start: continue
        tf_next_frame = oxt[i + 1]
        tf_between_frames = tf_velo_to_cam @ inv(tf_this_frame) @ tf_next_frame @ tf_cam_to_velo

        dx_gt.append(-tf_between_frames[0,3])
        dy_gt.append(-tf_between_frames[1,3])
        angles_gt.append(np.degrees(rotationMatrixToEulerAngles(tf_between_frames[0:4,0:4])[2]))

    for i, tf_this_frame in enumerate(est1[:-1]):
        if i == 0: continue
        tf_next_frame = est1[i + 1]
        tf_between_frames = tf_velo_to_cam @ inv(tf_this_frame) @ tf_next_frame @ tf_cam_to_velo

        dx1.append(tf_between_frames[0,3])
        dy1.append(tf_between_frames[1,3])
        angles1.append(np.degrees(rotationMatrixToEulerAngles(tf_between_frames[0:4,0:4])[2]))

    for i, tf_this_frame in enumerate(est2[:-1]):
        if i == 0: continue
        tf_next_frame = est2[i + 1]
        tf_between_frames = tf_velo_to_cam @ inv(tf_this_frame) @ tf_next_frame @ tf_cam_to_velo

        dx2.append(tf_between_frames[0,3])
        dy2.append(tf_between_frames[1,3])
        angles2.append(np.degrees(rotationMatrixToEulerAngles(tf_between_frames[0:4,0:4])[2]))



    print('\nOwn CNN:')

    print("Average X error:")
    print(np.abs(dx1-(-np.array(dx_gt))).mean())

    print("Average Y error:")
    print(np.abs(dy1-(-np.array(dy_gt))).mean())

    print("Average Angle error:")
    print(np.abs(angles1-(np.array(angles_gt))).mean())


    print('\nFlowNet:')

    print("Average X error:")
    print(np.abs(dx2-(-np.array(dx_gt))).mean())

    print("Average Y error:")
    print(np.abs(dy2-(-np.array(dy_gt))).mean())

    print("Average Angle error:")
    print(np.abs(angles2-(np.array(angles_gt))).mean())


    plt.subplot(311)
    plt.legend(handles=[mpatches.Patch(color='C0', label='OXT Ground Truth'), mpatches.Patch(color='C1', label='Own CNN'), mpatches.Patch(color='C2', label='FlowNet')])
    #plt.title('X-Bewegung')
    plt.plot(range(len(dx_gt)), -np.array(dx_gt), range(len(dx1)), dx1, range(len(dx2)), dx2)
    plt.subplot(312)
    plt.legend(handles=[mpatches.Patch(color='C0', label='OXT Ground Truth'), mpatches.Patch(color='C1', label='Own CNN'), mpatches.Patch(color='C2', label='FlowNet')])
    #plt.title('Y-Bewegung')
    plt.plot(range(len(dy_gt)), -np.array(dy_gt), range(len(dy1)), dy1, range(len(dy2)), dy2)
    plt.subplot(313)
    plt.legend(handles=[mpatches.Patch(color='C0', label='OXT Ground Truth'), mpatches.Patch(color='C1', label='Own CNN'), mpatches.Patch(color='C2', label='FlowNet')])
    #plt.title('Winkel')
    plt.plot(range(len(angles_gt)), np.array(angles_gt), range(len(angles1)), angles1, range(len(angles2)), angles2)
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
