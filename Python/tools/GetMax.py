import numpy as np
import cv2

filename = '/home/klein/U/gridmapsFlow/train/shuffled.csv'

with open(filename, 'r') as f:
    data_array = [line.strip().split(',') for line in f]

max_x = []
max_y = []
ln = len(data_array)

for line in data_array:
    layerX = (np.array(cv2.imread(line[2], cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128)[300:900, 300:900]
    layerY = (np.array(cv2.imread(line[3], cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128)[300:900, 300:900]

    max_x.append(np.amax(np.abs(layerX)))

    max_y.append(np.amax(np.abs(layerY)))

    print(len(max_x), ln)


print()

print(max_x)
print(max_y)