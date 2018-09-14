import sys, os
import numpy as np
import pcl
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from numpngw import write_png


def main(args):

    sequence = int(args[1])
    inpath = '/mrtstorage/projects/map_generation/kitti_map_2/'+str(sequence).zfill(2)+'/raw/'
    if sequence > 5: inpath = '/home/klein/U/kitti_pcds/'+str(sequence).zfill(2)+'/raw/'

    outpath = '/home/klein/U/XYZimage/kitti'+str(sequence).zfill(2)+'/'

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    with open(inpath+'map_el.txt', 'r') as f:
        data_array = np.array([line.strip().split(' ') for line in f])

    stamplist = [str(int(line)) for line in data_array[1:,1]]

    for i, timestamp in enumerate(stamplist):
        p = pcl.load(inpath+timestamp+'.pcd')
        pcd_array = np.asarray(p)

        img = np.zeros([45,628,4], dtype=np.float32)

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

            if 0 <= z < img.shape[0] and 0 <= int(phi*-100 + 314) < img.shape[1]:
                img[int(z)][int(phi*-100 + 314)] = np.append(point,r)

            if z > img.shape[0]: break

        # # fill in missing values:
        # invalid_cell_mask = np.isnan(img)
        # indices = nd.distance_transform_edt(invalid_cell_mask, return_distances=False, return_indices=True)
        # img = img[tuple(indices)]

        # convert to uint8:
        img = np.uint16(img*200 + 32768)

        write_png(outpath + 'xyz_16bit_'+timestamp+'.png', img)

        print('> ',i+1,'von',len(stamplist))

    return


def read():
    import tensorflow as tf
    image_str_ch0 = tf.read_file('/home/klein/U/XYZimage/kitti00/xyz_16bit_3421285.png')
    img = (tf.cast(tf.image.decode_png(image_str_ch0, channels=4, dtype=tf.uint16), dtype=tf.float32) - 32768)/200.0

    s = tf.Session()
    img = s.run(img)

    plt.subplot(411)
    a = plt.imshow(np.where(np.abs(img[:,:,0]) < 0.01, np.nan, img[:,:,0]))
    plt.colorbar(a, orientation='horizontal')
    plt.subplot(412)
    plt.imshow(np.where(np.abs(img[:,:,1]) < 0.01, np.nan, img[:,:,1]))
    plt.subplot(413)
    plt.imshow(np.where(np.abs(img[:,:,2]) < 0.01, np.nan, img[:,:,2]))
    plt.subplot(414)
    plt.imshow(img[:,:,3])
    plt.show()


if __name__ == "__main__":
    read()
    #main(sys.argv)
