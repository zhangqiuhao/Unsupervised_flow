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


def main(args):
    groundtruth_file = '/home/klein/U/gridmapsDynamic/train/kitti08/features.csv'
    in_folder = '/home/klein/U/inference/dynamic_seq_8'

    with open(groundtruth_file, 'r') as f:
        data_array = [[x if x.startswith('/') else float(x) for x in line.strip().split(',')] for line in f]

    epes = []

    for i, data_line in enumerate(data_array[:55]):
        if i < 11: continue

        fieldX = np.load(in_folder + '/' + str(i).zfill(4) + '_estimation_dX.npy')
        fieldY = np.load(in_folder + '/' + str(i).zfill(4) + '_estimation_dY.npy')

        fieldXgt = (np.array(cv2.imread(data_array[i][2], cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128)[300:900,300:900]
        fieldXgt = cv2.resize(fieldXgt, (300,300), cv2.INTER_NEAREST)
        fieldYgt = (np.array(cv2.imread(data_array[i][3], cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128)[300:900,300:900]
        fieldYgt = cv2.resize(fieldYgt, (300,300), cv2.INTER_NEAREST)

        mask = (np.array(cv2.imread(data_array[i][4], cv2.IMREAD_GRAYSCALE), dtype=np.float32))[300:900,300:900]
        mask = skimage.measure.block_reduce(mask, (2, 2), np.max)


        epe = np.nanmean(np.where(np.logical_and(fieldXgt != 0, mask == 255), np.sqrt(np.square(fieldX-fieldXgt) + np.square(fieldY-fieldYgt)), np.nan))
        epes.append(epe)


        fieldX = np.where(np.logical_and(fieldXgt != 0, mask == 255), fieldX, 0)
        fieldY = np.where(np.logical_and(fieldXgt != 0, mask == 255), fieldY, 0)
        flowfield = directionalField(fieldX, fieldY, 2)

        fieldXgt = np.where(np.logical_and(fieldXgt != 0, mask == 255), fieldXgt, 0)
        fieldYgt = np.where(np.logical_and(fieldXgt != 0, mask == 255), fieldYgt, 0)
        flowfieldgt = directionalField(fieldXgt, fieldYgt, 2)

        img = np.concatenate((fieldX, fieldXgt), axis=1)


        # flowfieldgt = directionalField(fieldXgt, fieldYgt, 2)
        # plt.imshow(img)
        # plt.show()

    print('Mean epe:', np.array(epes).mean())


if __name__ == '__main__':
    main(sys.argv)