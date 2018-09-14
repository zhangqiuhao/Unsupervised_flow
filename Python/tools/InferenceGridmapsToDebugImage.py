import numpy as np
from os import sys
sys.path.append('/home/klein/U/Masterarbeit/Python')
from tools.PlotDirectionalField import directionalField
import matplotlib.pyplot as plt
import cv2
import skimage.measure
from matplotlib.colors import LinearSegmentedColormap


def main(args):
    in_folder = '/home/klein/U/inference/gridmaps_seq_8/'
    gt_in = '/home/klein/U/gridmapsFlow_old/train/kitti08/features.csv'
    out_folder = '/home/klein/Desktop/U/sascha_praesentation_und_material/gridmap_error_seq_8/'

    np.set_printoptions(suppress=True)

    with open(gt_in, 'r') as f:
        gt_array = [[x.replace('gridmapsFlow/','gridmapsFlow_old/') if x.startswith('/') else float(x) for x in line.strip().split(',')] for line in f]

    for i in range(0,80):
        fieldX = np.load(in_folder + str(i).zfill(4) + '_estimation_dX.npy')
        fieldY = np.load(in_folder + str(i).zfill(4) + '_estimation_dY.npy')

        # fieldX = fieldX / 10
        # fieldY = fieldY / 10

        debug_img = directionalField(fieldX, fieldY, 8)

        # gt = cv2.imread(gt_in_folder + 'debug_' + str(i).zfill(6) + '.png', cv2.IMREAD_COLOR)
        # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        # gt = cv2.resize(gt, (300,300))

        fieldXgt = (np.array(cv2.imread(gt_array[i][2], cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128)[300:900,300:900]
        fieldXgt = cv2.resize(fieldXgt, (300,300), cv2.INTER_NEAREST)
        fieldYgt = (np.array(cv2.imread(gt_array[i][3], cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128)[300:900,300:900]
        fieldYgt = cv2.resize(fieldYgt, (300,300), cv2.INTER_NEAREST)

        gt = directionalField(fieldXgt, fieldYgt, 3)



        debug_img = np.where(gt == np.ones_like(gt), np.ones_like(gt), debug_img)


        # error image:
        cm = LinearSegmentedColormap.from_list('mylist', [(0, 1, 0), (1, 0, 0)], N=100)
        error_img = np.where(fieldXgt != 0, np.sqrt(np.square(fieldX-fieldXgt) + np.square(fieldY-fieldYgt)), np.nan)

        plt.imsave('/tmp/img.png', error_img, cmap=cm)
        error_img = plt.imread('/tmp/img.png')[:,:,0:3]

        error_img[:,:,0] = np.where(fieldXgt == 0, 1, error_img[:,:,0])
        error_img[:,:,1] = np.where(fieldXgt == 0, 1, error_img[:,:,1])
        error_img[:,:,2] = np.where(fieldXgt == 0, 1, error_img[:,:,2])

        concat_img = np.concatenate((debug_img, error_img), axis=1)

        plt.imsave(out_folder+ str(i).zfill(3) + '_debug.png', concat_img)

        print(i)


if __name__ == '__main__':
    main(sys.argv)