'''
Calculates depth images from pointcloud files. They have to be stored in mypath, where also a map_el.txt file has to
be present which holds the timestamps.
'''
import sys
import numpy as np
import pcl
import matplotlib.pyplot as plt
from scipy import ndimage as nd


def main(args):

    mypath = '/mrtstorage/projects/map_generation/kitti_map_2/00/raw/'

    with open(mypath+'map_el.txt', 'r') as f:
        data_array = np.array([line.strip().split(' ') for line in f])

    filelist = [mypath+str(int(line))+'.pcd' for line in data_array[1:,1]]

    for i, pcd_filename in enumerate(filelist):
        p = pcl.load(pcd_filename)
        pcd_array = np.asarray(p)

        dimg = np.zeros([64,628])
        dimg[:] = np.nan

        lastphi = 0
        z = 0

        for k, point in enumerate(pcd_array):
            r = np.sqrt(np.square(point[0])+np.square(point[1]))
            phi = np.arctan2(point[1], point[0])

            # detect line change
            if phi < 0 < lastphi:
                if np.sign(pcd_array[k:k + 20, 1]).sum() == -20:
                    z += 1
            lastphi = phi

            if 0 <= z < dimg.shape[0] and 0 <= int(phi*-100 + 314) < dimg.shape[1]:
                dimg[int(z)][int(phi*-100 + 314)] = r

            if z > dimg.shape[0]: break

        # fill in missing values:
        # invalid_cell_mask = np.isnan(dimg)
        # indices = nd.distance_transform_edt(invalid_cell_mask, return_distances=False, return_indices=True)
        # dimg = dimg[tuple(indices)]

        plt.imsave('/home/klein/d/testfile_'+str(i).zfill(3)+'.png',-dimg, cmap='Blues')

        print(i+1,'von',len(filelist))

    return




if __name__ == "__main__":
    main(sys.argv)
