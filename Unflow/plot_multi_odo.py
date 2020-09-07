import sys, os
import numpy as np
from numpy.linalg import inv
from math import degrees
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


np.set_printoptions(suppress=True)
sequence = '08'

########
start = 0
end = 4000
########
inpath_tf_oxt = '/home/zqhyyl/Masterarbeit/gridmap_train/08.txt'
networks = ['FN_4L_nophoto','FN_4L_gm','FN_4L_geo']

with open(inpath_tf_oxt, 'r') as f:
    oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]])))
           for line in f]
    points_oxt = [((l[0, 3], l[2, 3])) for l in oxt]
    points_oxt = points_oxt[start:end]
    path_oxt = Path(points_oxt, [Path.MOVETO] + [Path.LINETO] * (len(points_oxt) - 1))

odos = []

for network in networks:
    odo_save_path = '/home/zqhyyl/evaluation/odometry_' + network
    inpath_tf_is = odo_save_path + "/" + sequence + '_kitti.txt'
    with open(inpath_tf_is, 'r') as f:
        odo = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]
        points_estimation1 = [((l[0, 3], l[2, 3])) for l in odo]
        points_estimation1 = points_estimation1[0:end]
        path_estimation1 = Path(points_estimation1, [Path.MOVETO] + [Path.LINETO] * (len(points_estimation1)-1))
        odos.append(path_estimation1)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.grid(b=True, c='#cccccc')
handles=[patches.Patch(color='C0', label='OXTS Ground Truth')]
ax.add_patch(patches.PathPatch(path_oxt, facecolor='none', edgecolor='C0'))
for num, item in enumerate(odos):
    color = 'C'+str(num+1)
    handles = handles + [patches.Patch(color=color, label=networks[num])]
    ax.add_patch(patches.PathPatch(item, facecolor='none', edgecolor=color))
ax.legend(handles=handles)
ax.set_aspect('equal', 'datalim')
ax.set_xlabel('x in meter')
ax.set_ylabel('z in meter')

plt.autoscale(enable=True)
plt.savefig('/home/zqhyyl/evaluation/' + sequence + '_odometry.png')
