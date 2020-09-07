import os
import numpy as np


class Matrix():
    def __init__(self, dir):
        self.dir = dir

    def input_matrix(self):
        self._odometry_txt = open(self.dir).read()

    def return_matrix(self, num):
        self.input_matrix()
        return np.matrix(self._odometry_txt.split('\n')[num]).reshape(3,4)


def readlabel(num):
    label_path = '/mrtstorage/datasets/kitti/tracking/training/label_02/' + num + '.txt'
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
                object_indices = [1, 2]  # id, obj_type
                orientation_indices = [5, 10, 12, 13, 15, 16]  # obs_angle, w, l, X, Z, yaw
                if frame_num == last_frame:
                    list_temp.append(
                        list(parameters[object_indices]) + list(np.double(parameters[orientation_indices])))
                else:
                    data_list[last_frame] = list_temp
                    list_temp = []
                    list_temp.append(
                        list(parameters[object_indices]) + list(np.double(parameters[orientation_indices])))
                    last_frame = frame_num

    data_list[last_frame] = list_temp
    return data_list
