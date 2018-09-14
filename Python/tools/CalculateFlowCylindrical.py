import sys, os
import numpy as np
import pcl
import matplotlib.pyplot as plt
from PlotDirectionalField import directionalField
from scipy import ndimage as nd
from numpngw import write_png
import re
from math import radians, sin, cos
import shutil


def main(args):

    sequence = int(args[1])
    inpath_pcd = '/mrtstorage/projects/map_generation/kitti_map_2/'+str(sequence).zfill(2)+'/raw/'
    if sequence > 5: inpath_pcd = '/home/klein/U/kitti_pcds/'+str(sequence).zfill(2)+'/raw/'

    inpath_transforms = '/home/klein/U/gridmapsFlow/eval/kitti'+str(sequence).zfill(2)+'/features.csv'

    outpath = '/home/klein/U/groundtruthDebug/'

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    with open(inpath_transforms, 'r') as f:
        data_array = [[x if x.startswith('/') else float(x) for x in line.strip().split(',')] for line in f]


    for i, data_line in enumerate(data_array):

        stamp1 = re.search(r'prob_(\d+)', data_line[0]).group(1)
        stamp2 = re.search(r'prob_(\d+)', data_line[1]).group(1)

        p = pcl.load(inpath_pcd+stamp1+'.pcd')
        pcd_array = np.asarray(p)[:,0:2]

        img = np.zeros([45,628,3], dtype=np.float32)

        lastphi = 0
        z = 0

        for k, point in enumerate(pcd_array):
            phi = np.arctan2(point[1], point[0])

            # detect line change
            if phi < 0 < lastphi:
                if np.sign(pcd_array[k:k + 20, 1]).sum() == -20:
                    z += 1
            lastphi = phi

            if 0 <= z < img.shape[0] and 0 <= int(phi*-100 + 314) < img.shape[1]:
                # calculate x and y flow
                alpha = radians(data_line[-1])
                point_new = point @ np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]]) + np.array([data_line[-3]/10, data_line[-2]/10])
                img[int(z)][int(phi*-100 + 314)][0:2] = point_new - point

            if z > img.shape[0]: break



        # plt.subplot(311)
        # plt.imshow(img[:, :, 0])
        # plt.subplot(312)
        # plt.imshow(img[:, :, 1])
        # plt.subplot(313)
        # plt.imshow(directionalField(img[:, :, 0],img[:, :, 1]))
        # plt.show()

        plt.imsave(outpath + str(i).zfill(3) + '_gt_debug.png', directionalField(img[:, :, 0],img[:, :, 1], 1.3))

        # convert to uint8:
        img = np.uint16(img*2000 + 32768)

        print('> ',i+1,'von',len(data_array))

    return

if __name__ == "__main__":
    main(sys.argv)


