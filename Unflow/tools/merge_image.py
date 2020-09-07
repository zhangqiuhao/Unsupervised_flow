import cv2
import numpy as np
import os

filename_dir = '/home/zhang/pcl_data/08/data/'
flow_dir = '/home/zhang/ergebnisse/flow/'
out_dir = '/home/zhang/ergebnisse/flow_with_detections/'

number = 2
lst = os.listdir(flow_dir)
lst.sort()
for file in lst:
    filename = filename_dir + str(number).zfill(10) + '.png'
    number += 1

    img_base = cv2.imread(filename, 0)
    flow_img = cv2.imread(flow_dir + file, -1)

    print(filename_dir + file)

    pixelpoints = cv2.findNonZero(img_base)

    out_img = np.zeros(flow_img.shape, np.uint8)

    for points in pixelpoints:
        points = points[0]
        out_img[points[1], points[0]] = flow_img[points[1], points[0]]

    cv2.imwrite(out_dir + file, out_img)