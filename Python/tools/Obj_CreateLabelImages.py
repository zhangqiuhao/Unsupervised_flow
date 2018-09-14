# import xml.etree.ElementTree as ElementTree
# from xmlparse import XmlDictConfig
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys, os
import shutil
from os import listdir
from os.path import isfile, join
from numpy.linalg import inv
from math import sin, cos, tan

np.set_printoptions(suppress=True)

out_path = '/home/klein/U/object/'

if not os.path.exists(out_path):
    os.makedirs(out_path)

label_dir = '/home/klein/U/object/labels_txt/'
label_files = [join(label_dir, f) for f in listdir(label_dir) if isfile(join(label_dir, f))]
label_files.sort()

calib_dir = '/home/klein/U/object/calib/'


color = {'Car':(255,0,0), 'Pedestrian': (0,0,255), 'Cyclist': (0,255,0), 'DontCare':(0,0,0)}

for i, label_file in enumerate(label_files):
    if i == 144: continue

    with open(calib_dir + str(i).zfill(6) + '.txt', 'r') as f:
        data_calib = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(': ')[1].split(' ')], (3, 4)), [[0, 0, 0, 1]]))) if i == 5 else line for i, line in enumerate(f)]

    tf_cam_to_velo = inv(data_calib[5])
    tf_velo_to_cam = data_calib[5]

    # read label file:

    with open(label_file, 'r') as f:
        labels = [[item if i == 0 else float(item) for i, item in enumerate(line.strip().split())] for line in f]

    img = cv2.imread('/home/klein/U/object/gridmaps/0_euclidean_hits_'+str(i).zfill(6)+'.png', cv2.IMREAD_GRAYSCALE)
    labelimg = np.zeros([1200,1200])

    carFound = False
    for label in labels:
        if label[0] != 'Car': continue

        if label[0] == 'Car': carFound = True

        pos_cam = np.matrix(label[-4:-1] + [1]).T
        pos_velo = tf_cam_to_velo @ pos_cam

        x = 600 - int(pos_velo[1]*10)
        y = 600 - int(pos_velo[0]*10)
        l = int(label[-5] * 10 )
        w = int(label[-6] * 10 )

        box_raw = np.array([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 1, 1, 1, 1]])

        ry = -label[-1]
        shift_mat = np.array([[cos(ry), sin(ry), 0, x],
                              [-1 * sin(ry), cos(ry),0,y],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        box_shifted = shift_mat @ box_raw

        labelimg = cv2.fillPoly(labelimg, np.int32([np.array([[box_shifted[0,0],box_shifted[1,0]],[box_shifted[0,1],box_shifted[1,1]],[box_shifted[0,2],box_shifted[1,2]], [box_shifted[0,3],box_shifted[1,3]]])]), 1)


    if carFound:

        print('Write out', i)
        cv2.imwrite(out_path+'labels_img/'+str(i).zfill(6)+'.png', labelimg[0:600,300:900]*255)
        cv2.imwrite(out_path+'gridmaps_cropped/'+str(i).zfill(6)+'.png', 255 - img[0:600,300:900]*255)
