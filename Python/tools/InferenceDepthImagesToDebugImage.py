import numpy as np
from os import sys, path, makedirs
sys.path.append('/home/klein/U/Masterarbeit/Python')
from tools.PlotDirectionalField import directionalField
import matplotlib.pyplot as plt
import cv2
import skimage.measure
from matplotlib.colors import LinearSegmentedColormap


def main(args):


    in_folder = '/home/klein/U/inference/flownet_simple_seq_8/'
    gt_in = '/home/klein/U/depthimageFlow/train/kitti08/features.csv'
    out_folder = '/home/klein/U/debug/flownet_simple_error_seq_8/'

    if not path.exists(out_folder):
        makedirs(out_folder)

    np.set_printoptions(suppress=True)

    with open(gt_in, 'r') as f:
        gt_array = [[x for x in line.strip().split(',')] for line in f]

    for i in range(18,38):

        try:
            field = np.load(in_folder + str(i).zfill(3) + '_estimation.npy')
            fieldX = field[:, :, 0]
            fieldY = field[:, :, 1]
        except IOError:
            fieldX = np.load(in_folder + str(i).zfill(3) + '_estimation_dX.npy')
            fieldY = np.load(in_folder + str(i).zfill(3) + '_estimation_dY.npy')
            # fieldX = cv2.resize(fieldX, (1024,64))
            # fieldY = cv2.resize(fieldY, (1024,64))

        # field = np.load(in_folder + str(i).zfill(3) + '_estimation.npy')
        # fieldX = field[:, :, 0]
        # fieldY = field[:, :, 1]

        estimated_flow = directionalField(fieldX, fieldY, 2)

        image_gt = np.load(gt_array[i][2][:-3]+'npy')
        # image_gt = image_gt[0:45,:,:]
        image_gt = cv2.resize(image_gt, (1024,64))
        fieldXgt = image_gt[:, :, 0]
        fieldYgt = image_gt[:, :, 1]

        gt = directionalField(fieldXgt, fieldYgt, 2)

        estimated_flow = np.where(gt == np.ones_like(gt), np.ones_like(gt), estimated_flow)

        # error image:
        cm = LinearSegmentedColormap.from_list('mylist', [(0, 1, 0), (1, 0, 0)], N=100)
        error_img = np.where(fieldXgt != 0, np.sqrt(np.square(fieldX-fieldXgt) + np.square(fieldY-fieldYgt)), np.nan)

        plt.imsave('/tmp/img.png', error_img, cmap=cm)
        error_img = plt.imread('/tmp/img.png')[:,:,0:3]

        error_img[:,:,0] = np.where(fieldXgt == 0, 1, error_img[:,:,0])
        error_img[:,:,1] = np.where(fieldXgt == 0, 1, error_img[:,:,1])
        error_img[:,:,2] = np.where(fieldXgt == 0, 1, error_img[:,:,2])

        concat_img = np.concatenate((estimated_flow, error_img), axis=0)

        plt.imsave(out_folder+ str(i).zfill(3) + '_debug.png', concat_img)

        print(i)


if __name__ == '__main__':
    main(sys.argv)