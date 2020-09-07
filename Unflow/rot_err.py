import sys, os
import numpy as np
import math
from numpy.linalg import inv
from math import degrees
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

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
odo_save_path = '/home/zqhyyl/evaluation'

np.set_printoptions(suppress=True)

inpath_tf_oxt = '/home/zqhyyl/Masterarbeit/gridmap_train/08.txt'
inpath_tf_is = '/home/zqhyyl/evaluation/odometry_FN_4L_geo/08_odo.txt'
previous = 0
odo=[]

with open(inpath_tf_oxt, 'r') as f:
    oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]
with open(inpath_tf_is, 'r') as f:
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

plt.figure(figsize=(14,2))
plt.title("Vehicle rotation")
plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
plt.ylabel('Degree')
plt.legend(handles=[patches.Patch(color='C1', label='OXTS Ground Truth'), patches.Patch(color='C0', label='Unflow')])
plt.plot(timestamp, rot_oxt, 'r--', timestamp, odo, 'b--')
plt.savefig(odo_save_path + "/" + sequence + '_motion_rot.png',bbox_inches='tight')

timestamp = range(num-2)
plt.figure(figsize=(14,2))
plt.title("Vehicle rotation")
plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
plt.ylabel('Degree')
plt.legend(handles=[patches.Patch(color='C1', label='OXTS Ground Truth'), patches.Patch(color='C0', label='Unflow')])
plt.plot(timestamp, rot_rel_oxt, 'r--', timestamp, rot_rel_est, 'b--')
plt.savefig(odo_save_path + "/" + sequence + '_motion_rot_rel.png',bbox_inches='tight')


rot_err = rot_rel_est-rot_rel_oxt
plt.figure(figsize=(14,2))
plt.title("Vehicle rotation Error")
plt.xlabel('Timestamp', horizontalalignment='right', x=1.0)
plt.ylabel('Degree')
plt.legend(handles=[patches.Patch(label='Rot_err: {:.3f}'.format(np.mean(np.asarray(np.abs(rot_err)))))])
plt.plot(timestamp, rot_err, 'b--')
plt.savefig(odo_save_path + "/" + sequence + '_motion_rot_error.png',bbox_inches='tight')
