import os
import numpy as np

odometry_txt = open('/mrtstorage/datasets/kitti/odometry/data_odometry_groundtruth/dataset/poses/08.txt').read()

M = np.matrix(odometry_txt.split('\n')[1]).reshape(3,4)

print(M)
