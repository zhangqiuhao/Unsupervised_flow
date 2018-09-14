'''
Generates Scene Flow Maps from Gridmaps (generated by pointcloud_grid_map_tool)
and Kitti transformation data.
'''

import sys, os
import numpy as np
from numpy.linalg import inv
import cv2
from numpngw import write_png
import shutil
import matplotlib.pyplot as plt
from PlotDirectionalField import directionalField

def main(args):
    np.set_printoptions(suppress=True)

    sequence = int(args[1])

    inpath_gridmaps = '/home/klein/U/gridmapsNoTf/' + str(sequence).zfill(2) + '/'
    inpath_transforms = '/mrtstorage/datasets/kitti/odometry/all_py/poses/' + str(sequence).zfill(2) + '.txt'
    inpath_calib = '/mrtstorage/datasets/kitti/odometry/data_odometry_calib/sequences/' + str(sequence).zfill(2) + '/calib.txt'

    outpath = '/home/klein/U/tmp/train/kitti' + str(sequence).zfill(2) + '/'

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    with open(inpath_transforms, 'r') as f:
        transforms = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    with open(inpath_calib, 'r') as f:
        data_calib = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(': ')[1].split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    tf_cam_to_velo = data_calib[4]
    tf_velo_to_cam = inv(data_calib[4])

    with open(outpath + 'features.csv', 'w') as features:
        for i, tf_this_frame in enumerate(transforms):

            # calculate transform from this to next pointcloud:
            if i < len(transforms) - 1:
                tf_next_frame = transforms[i + 1]
            else:
                tf_next_frame = np.identity(4)

            tf_between_frames = tf_velo_to_cam @ inv(tf_this_frame) @ tf_next_frame @ tf_cam_to_velo

            gridmap = cv2.imread(inpath_gridmaps + '0_euclidean_hits_'+str(i).zfill(6)+'.png', cv2.IMREAD_GRAYSCALE)

            # mask pixels around car:
            cv2.rectangle(gridmap, (285,270), (315,315), 255, -1)

            flow = np.zeros(list(gridmap.shape) + [3], dtype=np.float32)

            indx, indy = np.where(gridmap < 255)
            for cnt,j in enumerate(indx):
                k = indy[cnt]
                # transform:
                point = np.matrix([(300 - j), (300 - k), 0, 1]).T
                point_new = tf_between_frames @ point
                flow[j][k] = (point-point_new).T[:, 0:3]

            # save debug image
            plt.imsave(outpath+'debug_' + str(i).zfill(6) + '.png', directionalField(flow[:, :, 0], flow[:, :, 1], 3))

            # convert to 16 bit
            flow = np.uint16(flow * 1000 + 32768)

            # copy gridmap file:
            cv2.imwrite(outpath + 'gridmap_'+str(i).zfill(6)+'.png', gridmap)

            if i < len(transforms) - 1:
                # write_png(outpath + 'flow_16bit_' + str(i).zfill(6) + '_' + str(i + 1).zfill(6) + '.png', flow)

                features.write(outpath + 'gridmap_' + str(i).zfill(6) + '.png' + ',' +
                               outpath + 'gridmap_' + str(i + 1).zfill(6) + '.png' + ',' +
                               outpath + 'flow_16bit_' + str(i).zfill(6) + '_' + str(i + 1).zfill(6) + '.png' + ',' +
                               str(tf_between_frames).replace('[', '').replace(']', '').replace('\n', '') + '\n')

                print('> ', i + 1, 'von', len(transforms))

    print('Done!')
    return


if __name__ == "__main__":
    main(sys.argv)


