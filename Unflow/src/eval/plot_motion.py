import sys, os
import numpy as np
import math
from numpy.linalg import inv
from math import degrees
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from .motion_evaluator import rotationMatrixToEulerAngles


def KittiPlotMotion(sequence, num, odo_save_path):
    np.set_printoptions(suppress=True)

    inpath_tf_oxt = '/home/zhang/pcl_data/gridmap_train/odo_gt/poses/'+sequence+'.txt'
    inpath_tf_is = odo_save_path + "/" + sequence + '_kitti.txt'

    with open(inpath_tf_oxt, 'r') as f:
     oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]
    with open(inpath_tf_is, 'r') as f:
     odo = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    points_oxt = []
    points_estimation1 = []
    rot_oxt = []
    rot_est = []
    for iter_ in range(num-1):
        l = oxt[iter_ + 1] - oxt[iter_]
        angle_matrix = np.linalg.inv(np.asarray(oxt[iter_][0:3, 0:3]))
        vec_rot = np.dot(angle_matrix, l[0:3, 3])
        points_oxt = points_oxt + [[vec_rot[0, 0], vec_rot[2, 0]]]
        rot_oxt = rot_oxt + [rotationMatrixToEulerAngles(oxt[iter_ + 1][0:3,0:3]) - rotationMatrixToEulerAngles(oxt[iter_][0:3,0:3])]

    for iter_ in range(num-1):
        l = odo[iter_ + 1] - odo[iter_]
        angle_matrix = np.linalg.inv(np.asarray(odo[iter_][0:3, 0:3]))
        vec_rot = np.dot(angle_matrix, l[0:3, 3])
        points_estimation1 = points_estimation1 + [[vec_rot[0, 0], vec_rot[2, 0]]]
        rot_est = rot_est + [rotationMatrixToEulerAngles(odo[iter_ + 1][0:3,0:3]) - rotationMatrixToEulerAngles(odo[iter_][0:3,0:3])]

    points_oxt = np.asarray(points_oxt)
    points_estimation1 = np.asarray(points_estimation1)
    rot_oxt = np.asarray(rot_oxt)/np.pi*180
    rot_est = np.asarray(rot_est)/np.pi*180

    timestamp = range(num-1)

    plt.figure(figsize=(10,4))
    plt.title("Vehicle rotation")
    plt.xlabel('Timestamp')
    plt.ylabel('Degree')
    plt.legend(handles=[patches.Patch(color='C1', label='OXTS Ground Truth'), patches.Patch(color='C0', label='Unflow')])
    plt.plot( timestamp, rot_oxt[:,1], 'r--', timestamp, rot_est[:,1], 'b--')
    plt.savefig(odo_save_path + "/" + sequence + '_motion_rot.png')

    plt.figure(figsize=(10,4))
    plt.title("Vehicle translation in X direction")
    plt.xlabel('Timestamp')
    plt.ylabel('Meter')
    plt.legend(handles=[patches.Patch(color='C1', label='OXTS Ground Truth'), patches.Patch(color='C0', label='Unflow')])
    plt.plot(timestamp, points_oxt[:,0], 'r--', timestamp, points_estimation1[:,0], 'b--')
    plt.savefig(odo_save_path + "/" + sequence + '_motion_X.png')

    plt.figure(figsize=(10,4))
    plt.xlabel('Timestamp')
    plt.ylabel('Meter')
    plt.title("Vehicle translation in Z direction")
    plt.legend(handles=[patches.Patch(color='C1', label='OXTS Ground Truth'), patches.Patch(color='C0', label='Unflow')])
    plt.plot(timestamp, points_oxt[:,1], 'r--', timestamp, points_estimation1[:,1], 'b--')
    plt.savefig(odo_save_path + "/" + sequence + '_motion_Z.png')

    #Errors
    rot_err = rot_oxt[:,1]-rot_est[:,1]
    plt.figure(figsize=(10,4))
    plt.title("Vehicle rotation Error")
    plt.xlabel('Timestamp')
    plt.ylabel('Degree')
    plt.legend(handles=[patches.Patch(label='Rot_err: '+str(np.mean(np.asarray(np.abs(rot_err)))))])
    plt.plot(timestamp, rot_err, 'b--')
    plt.savefig(odo_save_path + "/" + sequence + '_motion_rot_error.png')

    trans_x_err = points_oxt[:,0] - points_estimation1[:,0]
    plt.figure(figsize=(10,4))
    plt.title("Vehicle translation Error in X direction")
    plt.xlabel('Timestamp')
    plt.ylabel('Meter')
    plt.legend(handles=[patches.Patch(label='X_err: '+str(np.mean(np.asarray(np.abs(trans_x_err)))))])
    plt.plot(timestamp, trans_x_err, 'b--')
    plt.savefig(odo_save_path + "/" + sequence + '_motion_X_error.png')

    trans_z_err = points_oxt[:,1] - points_estimation1[:,1]
    plt.figure(figsize=(10,4))
    plt.xlabel('Timestamp')
    plt.ylabel('Meter')
    plt.title("Vehicle translation Error in Z direction")
    plt.legend(handles=[patches.Patch(label='Z_err: '+str(np.mean(np.asarray(np.abs(trans_z_err)))))])
    plt.plot(timestamp, trans_z_err, 'b--')
    plt.savefig(odo_save_path + "/" + sequence + '_motion_Z_error.png')