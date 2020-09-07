import cv2
import numpy as np
import os

data_set = '01'

filename_dir = '/home/zhang/pcl_data/' + data_set + '/gridmap/'
calib_dir = '/home/zhang/pcl_data/' + data_set + '/calib/'
dst_dir = '/home/zhang/pcl_data/' + data_set + '/data_occ/'

for file in os.listdir(calib_dir):
    filename = os.path.splitext(file)[0]

    filename_ground = filename_dir + filename + '_ground_surface_cartesian.png'
    filename_occ = filename_dir + filename + '_z_max_occlusions_cartesian.png'

    img_ground = cv2.imread(filename_ground, 0)
    img_occ = cv2.imread(filename_occ, 0)

    out_img = cv2.addWeighted(img_ground,1,img_occ,1,0)


    out_dir = dst_dir + filename + '.png'
    print(out_dir)

    cv2.imwrite(out_dir, out_img)
