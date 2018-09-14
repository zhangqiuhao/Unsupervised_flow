'''
Calculates X-Y-Flow gridmaps from existing globally labeled gridmaps.
Also copies the input files to new location.
'''
import cv2
import os
import shutil
import sys
sys.path.append('/home/klein/U/Masterarbeit/Python')
from tools import NumpyHelpers, PlotDirectionalField
from math import sin, cos, degrees, radians, acos
import numpy as np
import re

num = int(sys.argv[1])
in_file = '/home/klein/U/gridmapsfull/train/kitti'+str(num).zfill(2)+'/gridmaps.csv'
out_path = '/home/klein/U/gridmapsFlow/train/kitti'+str(num).zfill(2)

if not os.path.exists(out_path):
    os.makedirs(out_path)

with open(in_file, 'r') as f:
    data_array = [[x if x.startswith('/') else float(x) for x in line.strip().split(',')] for line in f]

with open(out_path+'/features.csv', 'w') as features:
    for i, data in enumerate(data_array):
        print(i+1, 'of', len(data_array))

        shutil.copy(data[0], out_path)
        shutil.copy(data[1], out_path)

        gridmap = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)

        tf_gridmaps = NumpyHelpers.XYAngleToMatrix(-data[-3], -data[-2], data[-1])

        x, y = gridmap.shape
        layerX = np.full([x, y], 128, dtype=np.float32)
        layerY = np.full([x, y], 128, dtype=np.float32)

        indx, indy = np.where(gridmap < 255)
        for cnt,j in enumerate(indx):
            k = indy[cnt]
            # transform:
            point = np.array([(600 - j), (600 - k)])
            phi = radians(data[-1])
            point_new = point @ np.array([[cos(phi), -sin(phi)],[sin(phi), cos(phi)]]) + np.array([-data[-3], -data[-2]])
            layerX[j][k] += (point_new[0]-point[0])
            layerY[j][k] += (point_new[1]-point[1])

        match = re.search(r'prob_(\d+)', data[0])
        timestamp0 = match.group(1)
        match = re.search(r'prob_(\d+)', data[1])
        timestamp1 = match.group(1)

        outfile_x = out_path + '/0_euclidean_vX_' + timestamp0 + '_' + timestamp1 + '.png'
        outfile_y = out_path + '/0_euclidean_vY_' + timestamp0 + '_' + timestamp1 + '.png'
        cv2.imwrite(outfile_x, layerX)
        cv2.imwrite(outfile_y, layerY)

        features.write(out_path+'/'+os.path.basename(data[0])+','+out_path+'/'+os.path.basename(data[1])+','+outfile_x+','+outfile_y+','+str(-data[-3])+','+str(-data[-2])+','+str(data[-1])+'\n')
