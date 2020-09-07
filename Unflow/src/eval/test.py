import numpy as np
import os
from PIL import Image, ImageDraw

num = '0000'
data_num = 0

label_path = '/mrtstorage/datasets/kitti/tracking/training/label_02/' + num + '.txt'
grid_map_path = '/home/zhang/pcl_data/' + num + '/gridmap/'
calib_path = '/home/zhang/pcl_data/' + num + '/calib/'

files = os.listdir(calib_path)
number_files = len(files)

data_list = [None] * number_files
list_temp = []
last_frame = 0

with open(label_path, 'r') as f:
    for line in f:
        parameters = np.asarray(line.strip().split(' '))
        frame_num = int(parameters[0])

        if parameters[2] != 'DontCare':
            object_indices = [1, 2]  # frame, id, obj_type
            orientation_indices = [5, 10, 12, 13, 15, 16]  # obs_angle, w, l, X, Z, yaw
            if frame_num == last_frame:
                list_temp.append(list(parameters[object_indices]) + list(np.double(parameters[orientation_indices])))
            else:
                data_list[last_frame] = list_temp
                list_temp = []
                list_temp.append(list(parameters[object_indices]) + list(np.double(parameters[orientation_indices])))
                last_frame = frame_num
data_list[last_frame] = list_temp

name = 0
for data in data_list:
    file_tmp = grid_map_path + str(name).zfill(6) + '_detections_cartesian.png'

    im = Image.open(file_tmp).convert('RGB')
    if data is not None:
        for object_data in data:
            mid = np.array((300 + object_data[5] / 0.1, 600 - object_data[6] / 0.1))

            obs_angle = object_data[2]
            yaw_angle = object_data[7]
            angle = -((np.pi + yaw_angle) + (np.pi + obs_angle))

            c, s = np.cos(angle), np.sin(angle)
            R = np.array(((c, s), (-s, c)))

            w = object_data[3] / 0.1 / 2.0
            l = object_data[4] / 0.1 / 2.0

            p_ul = R.dot(np.array(((-w), (-l)))) + mid
            p_ur = R.dot(np.array(((+w), (-l)))) + mid
            p_dl = R.dot(np.array(((-w), (+l)))) + mid
            p_dr = R.dot(np.array(((+w), (+l)))) + mid

            p_ul = [int(i) for i in p_ul]
            p_ur = [int(i) for i in p_ur]
            p_dl = [int(i) for i in p_dl]
            p_dr = [int(i) for i in p_dr]
            draw = ImageDraw.Draw(im)

            draw.line(p_ul+p_ur, fill=(255,0,0), width=3)
            draw.line(p_ur+p_dr, fill=(255,0,0), width=3)
            draw.line(p_dr+p_dl, fill=(255,0,0), width=3)
            draw.line(p_dl+p_ul, fill=(255,0,0), width=3)

    im.save('/home/zhang/test/' + str(name).zfill(6) + '.png')
    name += 1