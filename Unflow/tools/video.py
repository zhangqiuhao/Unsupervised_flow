import cv2
import numpy as np
import os

directory = '/home/zhang/flow_results/Tracking/fn/0007/'
list = os.listdir(directory)
number_files = len(list)
print number_files

img1 = cv2.imread(directory + '000003.png')
height, width, layers = img1.shape

video = cv2.VideoWriter('/home/zhang/video.avi', cv2.VideoWriter_fourcc(*"XVID"), 15, (width, height))

for i in range(number_files):
    filename = str(i)

    for j in range(6-len(str(i))):
        filename = '0' + filename

    file = directory + filename + '.png'

    print file

    img = cv2.imread(file)
    video.write(img)

video.release()
