import sys, os
import numpy as np
import math
from numpy.linalg import inv
from math import degrees
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
plt.rcParams['svg.fonttype'] = 'none'

# Checks if a matrix is a valid rotation matrix.
def _isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-5


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (_isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

num = 4000
sequence = '08'

np.set_printoptions(suppress=True)

network = 'FN'
inpath_tf_oxt = '/home/zqhyyl/pcl_data/poses/08.txt' 
inpath_tf_is = '/home/zqhyyl/evaluation/odometry_'+network+'/08.txt'
inpath_tf_angle = '/home/zqhyyl/evaluation/odometry_'+network+'/08_odo.txt'
odo_save_path = '/home/zqhyyl/evaluation/odometry_' + network

def rot_err():
    previous = 0
    odo=[]

    with open(inpath_tf_oxt, 'r') as f:
        oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]
    with open(inpath_tf_angle, 'r') as f:
        for line in f:
            this = float(line.strip().split(' ')[9])
            previous = this+previous
            odo = odo + [previous]

    rot_oxt = []
    previous = 0
    for iter_ in range(num-1):
        l = oxt[iter_+1] - oxt[iter_]
        angle = math.atan2(l[0,3], l[2,3])
        if abs(angle - previous) > np.pi/2:
            angle = angle + 2*np.pi
        previous = angle
        rot_oxt = rot_oxt + [angle]

    rot_oxt = np.asarray(rot_oxt)/np.pi*180
    odo = np.asarray(odo[0:-1])/np.pi*180

    rot_rel_oxt = []
    rot_rel_est = []
    for iter_ in range(num-2):
        rot_rel_est = rot_rel_est + [odo[iter_+1]-odo[iter_]]
        diff_oxt_rel = rotationMatrixToEulerAngles(oxt[iter_ + 1][0:3, 0:3]) - rotationMatrixToEulerAngles(oxt[iter_][0:3, 0:3])
        diff_oxt_rel = np.sign(odo[iter_+1]-odo[iter_])*np.abs(diff_oxt_rel[1])/np.pi*180
        rot_rel_oxt = rot_rel_oxt + [diff_oxt_rel]
    rot_rel_est = np.asarray(rot_rel_est)
    rot_rel_oxt = np.asarray(rot_rel_oxt)

    timestamp = range(num-1)

    plt.figure(figsize=(10,2), dpi=200)
    plt.title("Vehicle rotation")
    plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
    plt.ylabel('Degree')
    plt.plot(timestamp, rot_oxt, 'r', label='OXTS GT', linewidth=0.5)
    plt.plot(timestamp, odo, 'b', label='Unflow', linewidth=0.5)
    plt.legend(bbox_to_anchor=(0.98, 0.9), loc=1, borderaxespad=0.)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.85)
    plt.savefig(odo_save_path + "/" + sequence + '_motion_rot.svg', dpi=200)

    timestamp = range(num-2)
    plt.figure(figsize=(10,2), dpi=200)
    plt.title("Vehicle rotation")
    plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
    plt.ylabel('Degree')
    plt.plot(timestamp, rot_rel_oxt, 'r', label='OXTS GT', linewidth=0.5)
    plt.plot(timestamp, rot_rel_est, 'b', label='Unflow', linewidth=0.5)
    plt.legend(bbox_to_anchor=(0.98, 0.9), loc=1, borderaxespad=0.)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.85)
    plt.savefig(odo_save_path + "/" + sequence + '_motion_rot_rel.svg', dpi=200)


    rot_err = rot_rel_est-rot_rel_oxt
    abs_rot_err = np.abs(rot_err)
    mu_r = abs_rot_err.mean()
    textstr = r'$\mu=%.3f$' % (mu_r, )
    plt.figure(figsize=(10,2), dpi=200)
    plt.title("Absolute rotation Error")
    plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
    plt.ylabel('log(1+R_err) (Deg)')
    plt.plot(timestamp, np.log(1+abs_rot_err), 'k', linewidth=0.5)
    x0, xmax = plt.xlim()
    y0, ymax = plt.ylim()
    data_width = xmax - x0
    data_height = ymax - y0
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(x0 + data_width * 0.98, y0 + data_height * 0.9, textstr, fontsize=12, horizontalalignment='right', verticalalignment='top', bbox=props)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.85)
    plt.savefig(odo_save_path + "/" + sequence + '_motion_rot_error.svg', dpi=200)

    #histogram
    mu_rot = rot_err.mean()
    sigma_rot = rot_err.std()
    textstr = '\n'.join((r'$\mu=%.3f$' % (mu_rot, ), r'$\sigma=%.3f$' % (sigma_rot, )))
    plt.figure(figsize=(5,5), dpi=200)
    plt.title("Histogram of Rotation Error")
    plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
    plt.ylabel('Frequecy')
    num_bins = 40
    n, bins, patches = plt.hist(rot_err, num_bins, facecolor='blue', alpha=0.5)
    x0, xmax = plt.xlim()
    y0, ymax = plt.ylim()
    data_width = xmax - x0
    data_height = ymax - y0
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(x0 + data_width * 0.98, y0 + data_height * 0.9, textstr, fontsize=12, horizontalalignment='right', verticalalignment='top', bbox=props)
    plt.savefig(odo_save_path + "/" + sequence + '_Rot_error_hist.svg', dpi=200)


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
 rot_oxt = rot_oxt + [rotationMatrixToEulerAngles(oxt[iter_][0:3,0:3]) - rotationMatrixToEulerAngles(oxt[iter_][0:3,0:3])]

for iter_ in range(num-1):
 l = odo[iter_ + 1] - odo[iter_]
 angle_matrix = np.linalg.inv(np.asarray(odo[iter_][0:3, 0:3]))
 vec_rot = np.dot(angle_matrix, l[0:3, 3]/600*512)
 points_estimation1 = points_estimation1 + [[vec_rot[0, 0], vec_rot[2, 0]]]
 rot_est = rot_est + [rotationMatrixToEulerAngles(odo[iter_][0:3,0:3]) - rotationMatrixToEulerAngles(odo[iter_][0:3,0:3])]

points_oxt = np.asarray(points_oxt)
points_estimation1 = np.asarray(points_estimation1)
rot_oxt = np.asarray(rot_oxt)/np.pi*180
rot_est = np.asarray(rot_est)/np.pi*180

timestamp = range(num-1)

plt.figure(figsize=(10,2), dpi=200)
plt.title("Vehicle translation in X direction")
plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
plt.ylabel('Meter')
plt.plot(timestamp, points_oxt[:,0], 'r', label='OXTS GT', linewidth=0.5)
plt.plot(timestamp, points_estimation1[:,0], 'b', label='Unflow', linewidth=0.5)
plt.legend(bbox_to_anchor=(0.98, 0.9), loc=1, borderaxespad=0.)
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.85)
plt.savefig(odo_save_path + "/" + sequence + '_motion_X.svg', dpi=200)

plt.figure(figsize=(10,2), dpi=200)
plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
plt.ylabel('Meter')
plt.title("Vehicle translation in Z direction")
plt.plot(timestamp, points_oxt[:,1], 'r', label='OXTS GT', linewidth=0.5)
plt.plot(timestamp, points_estimation1[:,1], 'b', label='Unflow', linewidth=0.5)
plt.legend(bbox_to_anchor=(0.98, 0.9), loc=1, borderaxespad=0.)
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.85)
plt.savefig(odo_save_path + "/" + sequence + '_motion_Z.svg', dpi=200)

#Errors

x_err = points_estimation1[:,0] - points_oxt[:,0]
trans_x_err = np.abs(x_err)
mu_x = trans_x_err.mean()
textstr = r'$\mu=%.4f$' % (mu_x, )
plt.figure(figsize=(10,2), dpi=200)
plt.title("Absolute translation error in X direction")
plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
plt.ylabel('log(1+X_err) (m)')
plt.plot(timestamp, np.log(1+trans_x_err), 'k', linewidth=0.5)
x0, xmax = plt.xlim()
y0, ymax = plt.ylim()
data_width = xmax - x0
data_height = ymax - y0
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(x0 + data_width * 0.98, y0 + data_height * 0.9, textstr, fontsize=12, horizontalalignment='right', verticalalignment='top', bbox=props)
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.85)
plt.savefig(odo_save_path + "/" + sequence + '_motion_X_error.svg', dpi=200)

#histogram
mu_x = x_err.mean()
sigma_x = x_err.std()
textstr = '\n'.join((r'$\mu=%.4f$' % (mu_x, ), r'$\sigma=%.4f$' % (sigma_x, )))
plt.figure(figsize=(5,5), dpi=200)
plt.title("Histogram of translation Error in X direction")
plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
plt.ylabel('Frequecy')
num_bins = 40
n, bins, patches = plt.hist(x_err, num_bins, facecolor='blue', alpha=0.5)
x0, xmax = plt.xlim()
y0, ymax = plt.ylim()
data_width = xmax - x0
data_height = ymax - y0
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(x0 + data_width * 0.98, y0 + data_height * 0.9, textstr, fontsize=12, horizontalalignment='right', verticalalignment='top', bbox=props)
plt.savefig(odo_save_path + "/" + sequence + '_X_error_hist.svg', dpi=200)

z_err = points_estimation1[:,1] - points_oxt[:,1]
trans_z_err = np.abs(z_err)
mu_z = trans_z_err.mean()
textstr = r'$\mu=%.3f$' % (mu_z, )
plt.figure(figsize=(10,2), dpi=200)
plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
plt.ylabel('log(1+Z_err) (m)')
plt.title("Absolute translation Error in Z direction")
plt.plot(timestamp, np.log(1+trans_z_err), 'k', linewidth=0.5)
x0, xmax = plt.xlim()
y0, ymax = plt.ylim()
data_width = xmax - x0
data_height = ymax - y0
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(x0 + data_width * 0.98, y0 + data_height * 0.9, textstr, fontsize=12, horizontalalignment='right', verticalalignment='top', bbox=props)
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.85)
plt.savefig(odo_save_path + "/" + sequence + '_motion_Z_error.svg', dpi=200)

#histogram
mu_z = z_err.mean()
sigma_z = z_err.std()
textstr = '\n'.join((r'$\mu=%.3f$' % (mu_z, ), r'$\sigma=%.3f$' % (sigma_z, )))
plt.figure(figsize=(5,5), dpi=200)
plt.title("Histogram of translation Error in Z direction")
plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
plt.ylabel('Frequecy')
num_bins = 40
n, bins, patches = plt.hist(z_err, num_bins, facecolor='blue', alpha=0.5)
x0, xmax = plt.xlim()
y0, ymax = plt.ylim()
data_width = xmax - x0
data_height = ymax - y0
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(x0 + data_width * 0.98, y0 + data_height * 0.9, textstr, fontsize=12, horizontalalignment='right', verticalalignment='top', bbox=props)
plt.savefig(odo_save_path + "/" + sequence + '_Z_error_hist.svg', dpi=200)

rot_err()
