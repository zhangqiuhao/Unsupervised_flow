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


sequence = ['00','01','02','03','04','05','06','07','08','09','10']
np.set_printoptions(suppress=True)
points_oxt = []
rot_oxt = []

for seq in sequence:
    inpath_tf_oxt = '/home/zhang/pcl_data/gridmap_train/odo_gt/poses/'+ seq + '.txt'

    with open(inpath_tf_oxt, 'r') as f:
        oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]
    num = len(oxt)
    for iter_ in range(num-1):
        l = oxt[iter_ + 1] - oxt[iter_]
        points_oxt = points_oxt + [[l[0,3], l[2,3]]]
        rot_oxt = rot_oxt + [rotationMatrixToEulerAngles(oxt[iter_ + 1][0:3,0:3]) - rotationMatrixToEulerAngles(oxt[iter_][0:3,0:3])]

points_oxt = np.asarray(points_oxt)*512/60
rot_oxt = np.asarray(rot_oxt)/np.pi*180
print(len(rot_oxt))

n, bins, patches = plt.hist(x=rot_oxt[:,1], bins='auto', color='#0504aa')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Rotation in °')
plt.ylabel('Frequency')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()


