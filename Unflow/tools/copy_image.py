from shutil import copyfile
import os

train_number = ['0000','0002','0003','0004','0005','0006','0007','0008','0009','0010','08','09','10','11','12','13','14','15']
directory = '/home/zhang/pcl_data/gridmap_train/evaluate/'
file_names = ['_intensity_cartesian', '_decay_rate_cartesian', '_detections_cartesian',
             '_observations_cartesian', '_z_max_detections_cartesian', '_z_max_occlusions_cartesian',
             '_z_min_detections_cartesian', '_z_min_observations_cartesian']
for num in train_number:
    print(num)
    for file_name in file_names:
        src = directory + num + '/gridmap/000000' + file_name + '.png'
        dst_1 = directory + num + '/gridmap/0000000' + file_name + '.png'
        dst_2 = directory + num + '/gridmap/00000000' + file_name + '.png'
        copyfile(src, dst_1)
        copyfile(src, dst_2)
