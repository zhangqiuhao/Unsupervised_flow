import cv2
import numpy as np
import os

train_number = '13'

filename_dir = '/home/zhang/pcl_data/' + train_number + '/calib/'
directory = '/home/zhang/pcl_data/' + train_number + '/gridmap/'
dst_dir = '/home/zhang/pcl_data/' + train_number + '/data/'

for file in os.listdir(filename_dir):
    filename = os.path.splitext(file)[0]

    img_1_ = directory + filename + '_decay_rate_cartesian.png'
    img_2_ = directory + filename + '_z_max_detections_cartesian.png'
    img_3_ = directory + filename + '_z_min_detections_cartesian.png'
    img_4_ = directory + filename + '_intensity_cartesian.png'
    img_5_ = directory + filename + '_rgb_cartesian.png'

    mask_img_ = directory + filename + '_detections_cartesian.png'

    img1 = cv2.imread(img_1_, 0)
    img2 = cv2.imread(img_2_, 0)
    img3 = cv2.imread(img_3_, 0)
    img4 = cv2.imread(img_4_, 0)
    img5 = cv2.imread(img_5_, 1)

    mask_img = cv2.imread(mask_img_, 0)

    pixelpoints = cv2.findNonZero(mask_img)

    img_1_out = np.zeros(img1.shape, np.uint8)
    img_2_out = np.zeros(img2.shape, np.uint8)
    img_3_out = np.zeros(img3.shape, np.uint8)
    img_4_out = np.zeros(img4.shape, np.uint8)
    img_5_out = np.zeros(img5.shape, np.uint8)

    for points in pixelpoints:
        points = points[0]
        img_1_out[points[1], points[0]] = img1[points[1], points[0]]
        img_2_out[points[1], points[0]] = img2[points[1], points[0]]
        img_3_out[points[1], points[0]] = img3[points[1], points[0]]
        img_4_out[points[1], points[0]] = img4[points[1], points[0]]
        img_5_out[points[1], points[0]] = img5[points[1], points[0]]

    #output = cv2.merge((img_1_out, img_2_out, img_3_out))

    #out_dir = dst_dir + '0000' + filename + '.png'

    print(filename)

    cv2.imwrite(img_1_, img_1_out)
    cv2.imwrite(img_2_, img_2_out)
    cv2.imwrite(img_3_, img_3_out)
    cv2.imwrite(img_4_, img_4_out)
    cv2.imwrite(img_5_, img_5_out)
