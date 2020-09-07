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
########
inpath_tf_oxt = '/home/zqhyyl/pcl_data/poses/'+sequence+'.txt'

with open(inpath_tf_oxt, 'r') as f:
    oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]])))
           for line in f]
    points_oxt = [((l[0, 3], l[2, 3])) for l in oxt]
    end = len(points_oxt)
    points_oxt = points_oxt[start:end]
    path_oxt = Path(points_oxt, [Path.MOVETO] + [Path.LINETO] * (len(points_oxt) - 1))

odos = []

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.grid(b=True, c='#cccccc')
handles=[patches.Patch(color='C0', label='OXTS Ground Truth')]
ax.add_patch(patches.PathPatch(path_oxt, facecolor='none', edgecolor='C0'))
ax.legend(handles=handles)
ax.set_aspect('equal', 'datalim')
ax.set_xlabel('x in meter')
ax.set_ylabel('z in meter')

plt.autoscale(enable=True)
plt.savefig('/home/zqhyyl/'+sequence+'odometry.png')
