'''
Generates cylindrical 3-Channel Depth Images and their Scene Flow from
Kitti Data.
'''

import sys, os
import numpy as np
from numpy.linalg import inv
import pcl
from numpngw import write_png
from math import degrees
import matplotlib.pyplot as plt
from PlotDirectionalField import directionalField


def main(args):
    np.set_printoptions(suppress=True)

    sequence = int(args[1])

    inpath_pcd = '/mrtstorage/datasets/kitti/odometry/data_odometry_velodyne_pcd/' + str(sequence).zfill(2) + '/pcds/'
    inpath_transforms = '/mrtstorage/datasets/kitti/odometry/all_py/poses/' + str(sequence).zfill(2) + '.txt'
    inpath_calib = '/mrtstorage/datasets/kitti/odometry/data_odometry_calib/sequences/' + str(sequence).zfill(2) + '/calib.txt'

    outpath = '/home/klein/U/depthimageFlow/train/kitti' + str(sequence).zfill(2) + '/'

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

            p = pcl.load(inpath_pcd + str(i).zfill(6) + '.pcd')
            pcd_array = np.asarray(p)

            depth_img = np.zeros([64, 4 * 360, 3], dtype=np.float32)
            flow_img = np.zeros([64, 4 * 360, 3], dtype=np.float32)

            lastphi = 0
            z = 0
            for k, point in enumerate(pcd_array):
                phi = np.arctan2(point[1], point[0])

                # detect line change
                if phi < 0 < lastphi:
                    if np.sign(pcd_array[k:k + 20, 1]).sum() == -20:
                        z += 1
                lastphi = phi

                u = int(degrees(phi) * -4 + 4*180)
                if 0 <= z < depth_img.shape[0] and 0 <= u < depth_img.shape[1]:
                    depth_img[z][u] = point
                    # calculate flow
                    point = np.matrix(np.append(point, 1)).T
                    point_new = tf_between_frames @ point
                    flow_img[z][u] = (point-point_new).T[:, 0:3]

                if z > depth_img.shape[0]: break

            # save debug image
            plt.imsave(outpath+'debug_' + str(i).zfill(6) + '.png', directionalField(flow_img[:, :, 0], flow_img[:, :, 1]))

            # convert to uint16:
            depth_img = np.uint16(depth_img * 200 + 32768)
            flow_img = np.uint16(flow_img * 2000 + 32768)

            write_png(outpath + 'xyz_16bit_' + str(i).zfill(6) + '.png', depth_img)
            if i < len(transforms) - 1:
                write_png(outpath + 'flow_16bit_' + str(i).zfill(6) + '_' + str(i + 1).zfill(6) + '.png', flow_img)

                features.write(outpath + 'xyz_16bit_' + str(i).zfill(6) + '.png' + ',' +
                               outpath + 'xyz_16bit_' + str(i + 1).zfill(6) + '.png' + ',' +
                               outpath + 'flow_16bit_' + str(i).zfill(6) + '_' + str(i + 1).zfill(6) + '.png' + ',' +
                               str(tf_between_frames).replace('[', '').replace(']', '').replace('\n', '') + '\n')

                print('> ', i + 1, 'von', len(transforms))

    print('Done!')
    return


if __name__ == "__main__":
    main(sys.argv)


