import numpy as np
import svgwrite
from os import walk, path, sys
sys.path.append('/home/klein/U/Masterarbeit/Python')
from math import sin, cos, radians
from tools.NumpyHelpers import XYAngleToMatrix2, similarity_transform
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi, degrees
import skimage.measure
from tools.PlotDirectionalField import directionalField
from matplotlib.colors import LinearSegmentedColormap


def checkForDynamicObjects(img0, img1, flow0, flow1):
    map0 = (255-np.array(img0, dtype=np.float32))[300:900,300:900]
    map1 = (255-np.array(img1, dtype=np.float32))[300:900,300:900]

    fieldXgt = (np.array(flow0, dtype=np.float32) - 128)[300:900,300:900]
    fieldYgt = (np.array(flow1, dtype=np.float32) - 128)[300:900,300:900]

    new_map0 = np.zeros_like(map0)

    indx, indy = np.where(map0 != 0)
    for cnt,j in enumerate(indx):
        k = indy[cnt]

        if 0 < int(j - fieldXgt[j,k]) < 600 and 0 < int(k - fieldYgt[j,k]) < 600:
            new_map0[int(j - fieldXgt[j,k]),int(k - fieldYgt[j,k])] = map0[j,k]


    _, new_map0 = cv2.threshold(new_map0, 1, 255, cv2.THRESH_BINARY)
    new_map0 = cv2.dilate(new_map0, np.ones((3,3)), iterations=1)

    _, map1 = cv2.threshold(map1, 1, 255, cv2.THRESH_BINARY)
    map1 = cv2.dilate(map1, np.ones((6,6)), iterations=1)

    diff = np.where(map1 - new_map0 < 0, np.ones_like(map0, dtype=np.uint8), np.zeros_like(map0, dtype=np.uint8))
    diff = cv2.erode(diff, np.ones((2, 2)), iterations=4)


    filter = np.array(cv2.circle(np.zeros((5,5)), (2,2), 2, 1 , -1),dtype=np.uint8)
    diff = cv2.dilate(diff, filter, iterations=3)

    cimg = 255-cv2.cvtColor(np.array(map0,dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    diff = np.array(diff,dtype=np.uint8)
    overlay = np.ones_like(cimg)*255
    overlay[:,:,1] = 255-diff*255
    cimg = cv2.addWeighted(overlay, .5, cimg, 0.5, 0)

    return cimg


def main(args):
    datafile = '/home/klein/U/gridmapsFlow/train/kitti09/features.csv'

    np.set_printoptions(suppress=True)

    with open(datafile, 'r') as f:
        data_array = [[x if x.startswith('/') else float(x) for x in line.strip().split(',')] for line in f]


    filtersize = 4
    i = 0

    for data_line in data_array:
        img0 = cv2.imread(data_line[0], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(data_line[1], cv2.IMREAD_GRAYSCALE)

        flow0 = cv2.imread(data_line[2], cv2.IMREAD_GRAYSCALE)
        flow1 = cv2.imread(data_line[3], cv2.IMREAD_GRAYSCALE)

        cimg = checkForDynamicObjects(img0, img1, flow0, flow1)

        plt.imsave('/home/klein/U/tmp_dynamic/'+str(i).zfill(4)+'_dynamic.png', cimg)
        print(i)
        i += 1



if __name__ == '__main__':
    main(sys.argv)