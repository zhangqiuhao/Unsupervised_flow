import cv2
import numpy as np
import os

train_numbers = ['00', '01', '02', '03', '04', '05', '06', '07']

min=512*512
count = 0
count_file = 0

for train_number in train_numbers:
    filename_dir = '/home/zhang/pcl_data/' + train_number + '/calib/'
    directory = '/home/zhang/pcl_data/' + train_number + '/gridmap/'

    for file in os.listdir(filename_dir):
        filename = os.path.splitext(file)[0]

        img = directory + filename + '_detections_cartesian.png'

        img = cv2.imread(img, 0)

        pixelpoints = cv2.findNonZero(img)
        num = pixelpoints.shape[0]
        count += num
        count_file += 1
        if num < min:
            min = num
            print(min, directory + filename)

print(count / count_file)
