import numpy as np
from os import sys
from math import sin, cos, radians, degrees
from numpy.linalg import inv
from NumpyHelpers import XYAngleToMatrix2, similarity_transform
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def main(args):

    sequence = 9

    groundtruth_file = '/home/klein/U/depthimageFlow/eval/kitti' + str(sequence).zfill(2) + '/features.csv'
    inpath_calib = '/mrtstorage/datasets/kitti/odometry/data_odometry_calib/sequences/' + str(sequence).zfill(
        2) + '/calib.txt'

    inpath_tf_oxt = '/mrtstorage/datasets/kitti/odometry/all_py/poses/' + str(sequence).zfill(2) + '.txt'
    with open(inpath_tf_oxt, 'r') as f:
        oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    with open(inpath_calib, 'r') as f:
        data_calib = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(': ')[1].split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    tf_cam_to_velo = data_calib[4]
    tf_velo_to_cam = inv(data_calib[4])

    ##################
    in_folder = '/home/klein/U/inference/flownet_simple_seq_9/'
    start = 500
    end = 800
    ##################

    np.set_printoptions(suppress=True)

    with open(groundtruth_file, 'r') as f:
        data_array = [[x for x in line.strip().split(',')] for line in f]

    epes = []

    if start == 0:
        lastposition = np.identity(4)
    else:
        lastposition = oxt[start-1]

    with open(in_folder + 'estimation_kitti_poses.txt', 'w') as kittifile:
        kittifile.write('1 0 0 0 0 1 0 0 0 0 1 0\n')

        for i, data_line in enumerate(data_array[:end]):
            if i < start: continue

            try:
                field = np.load(in_folder + str(i).zfill(3) + '_estimation.npy')
                fieldX = field[:, :, 0]
                fieldY = field[:, :, 1]
            except IOError:
                fieldX = np.load(in_folder + str(i).zfill(3) + '_estimation_dX.npy')
                fieldY = np.load(in_folder + str(i).zfill(3) + '_estimation_dY.npy')
                field = np.stack([fieldX, fieldY, np.zeros_like(fieldX)], axis=2)

            fieldX = cv2.resize(fieldX, (1024, 64))
            fieldY = cv2.resize(fieldY, (1024, 64))
            field = cv2.resize(field, (1024, 64))

            image = np.load(data_line[0][:-3]+'npy')
            image = cv2.resize(image, (1024,64))
            layerX = image[:, :, 0]
            layerY = image[:, :, 1]

            image_gt = np.load(data_line[2][:-3]+'npy')
            image_gt = cv2.resize(image_gt, (1024,64))
            fieldXgt = image_gt[:, :, 0]
            fieldYgt = image_gt[:, :, 1]

            # create mask
            mask = np.where(layerX != 0, 1, 0)
            kernel = np.ones((4, 4), np.uint8)
            mask = cv2.erode(mask.astype(np.float32),kernel,iterations = 1)

            epe = np.nanmean(np.where(mask, np.sqrt(np.square(fieldX-fieldXgt) + np.square(fieldY-fieldYgt)), np.nan))

            print(epe)
            epes.append(epe)


            # plt.subplot(711)
            # plt.title('Estimation')
            # plt.imshow(fieldX)
            #
            # plt.subplot(712)
            # plt.title('Ground Truth')
            # plt.imshow(fieldXgt)
            #
            # plt.subplot(713)
            # plt.title('Error')
            # cm = LinearSegmentedColormap.from_list('mylist', [(0, 1, 0), (1, 0, 0)], N=100)
            # plt.imshow(np.abs(np.where(layerX != 0,fieldXgt,np.nan) - np.where(layerX != 0,fieldX,np.nan)), cmap=cm)
            #
            # plt.subplot(714)
            # plt.title('Mask')
            # plt.imshow(mask, cmap='Greys')
            #
            # plt.subplot(715)
            # plt.title('Masked Estimation')
            # # plt.gca().set_facecolor('black')
            # plt.imshow(np.where(mask,fieldX,np.nan))
            #
            # plt.subplot(716)
            # plt.title('Masked Ground Truth')
            # # plt.gca().set_facecolor('black')
            # plt.imshow(np.where(mask,fieldXgt,np.nan))
            #
            # plt.subplot(717)
            # plt.title('Error')
            # cm = LinearSegmentedColormap.from_list('mylist', [(0, 1, 0), (1, 0, 0)], N=100)
            # plt.imshow(np.abs(np.where(mask,fieldXgt,np.nan) - np.where(mask,fieldX,np.nan)), cmap=cm)
            #
            # plt.show()

            # combine field estimation to one single gridmap estimation
            points1X = np.where(mask, image[:, :, 0], np.nan)
            points1Y = np.where(mask, image[:, :, 1], np.nan)
            points1Z = np.where(mask, image[:, :, 2], np.nan)
            points2X = np.where(mask, image[:, :, 0] - field[:, :, 0], np.nan)
            points2Y = np.where(mask, image[:, :, 1] - field[:, :, 1], np.nan)
            points2Z = np.where(mask, image[:, :, 2] - field[:, :, 2], np.nan)

            points1X = points1X[~np.isnan(points1X)]
            points1Y = points1Y[~np.isnan(points1Y)]
            points1Z = points1Z[~np.isnan(points1Z)]
            points2X = points2X[~np.isnan(points2X)]
            points2Y = points2Y[~np.isnan(points2Y)]
            points2Z = points2Z[~np.isnan(points2Z)]

            points1 = np.stack((points1X, points1Y, points1Z),axis=1)
            points2 = np.stack((points2X, points2Y, points2Z), axis=1)

            R, t = similarity_transform(points1, points2)
            tf_estimation = np.concatenate((np.concatenate((R,np.matrix(t).T), axis=1),[[0,0,0,1]]), axis=0)

            # transform transform to camera coordinates:
            tf_estimation = tf_cam_to_velo @ tf_estimation @ tf_velo_to_cam

            # join transforms:
            lastposition = lastposition @ tf_estimation
            kittifile.write(' '.join(str(lastposition[0:3,0:4].flatten()[0]).strip('[] ').split()) + '\n')



    print("Final EPE:")
    print(np.nanmean(np.array(epes)))



if __name__ == '__main__':
    main(sys.argv)