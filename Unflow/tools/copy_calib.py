from shutil import copyfile
import os

train_number = '15'
directory = '/mrtstorage/datasets/kitti/odometry/data_odometry_velodyne_pcd/'+train_number+'/pcds'
src = '/home/zhang/pcl_data/' + train_number + '/calib.txt'
for filename in os.listdir(directory):
    filename_ = os.path.splitext(filename)[0]
    dst = '/home/zhang/pcl_data/' + train_number + '/calib/' + filename_ + '.txt'
    print(dst)
    copyfile(src, dst)
