import numpy as np
from PlotDirectionalField import directionalField
import matplotlib.pyplot as plt
import cv2

inpath = '/home/klein/U/inference/gridmaps_seq_0/'

for i in range(0,250):

    try:
        field = np.load(inpath + str(i).zfill(3) + '_estimation.npy')
        layerX = field[:, :, 0]
        layerY = field[:, :, 1]
    except IOError:
        layerX = np.load(inpath + str(i).zfill(4) + '_estimation_dX.npy')
        layerY = np.load(inpath + str(i).zfill(4) + '_estimation_dY.npy')

    field = directionalField(layerX, layerY, 3)

    plt.imsave(inpath + str(i).zfill(3) + '_debug.png', field)

    print(i)


# outpath = '/home/klein/U/groundtruthDebug_seq9/'
#
# groundtruth_file = '/home/klein/U/xyzdFlow/eval/kitti00/features.csv'
#
# with open(groundtruth_file, 'r') as f:
#     data_array = [[x if x.startswith('/') else float(x) for x in line.strip().split(',')] for line in f]
#
# for i, data_line in enumerate(data_array[:200]):
#
#     image_gt = np.load(data_line[2][:-3] + 'npy')
#     layerX = image_gt[:, :, 0]
#     layerY = image_gt[:, :, 1]
#
#     field = directionalField(layerX, layerY, 1.3)
#
#     plt.imsave(outpath + str(i).zfill(3) + '_gt_debug.png', field)

#
# inpath = '/home/klein/Desktop/sascha_bilder/'
#
# for i in range(200):
#     estimation = np.array(plt.imread(inpath + str(i).zfill(3) + '_debug.png'))
#     gt = np.array(plt.imread(inpath + str(i).zfill(3) + '_gt_debug.png'))
#
#     con = np.concatenate((gt,estimation),axis=0)
#
#     con = cv2.putText(con, "Ground Truth", (8,12), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0,1))
#     con = cv2.putText(con, "CNN Estimation", (8, 12+45), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0, 1))
#
#     plt.imsave(inpath + str(i).zfill(3) + '_combination.png', con)


#
# inpath_gt = '/home/klein/U/depthimageFlow/eval/kitti09/'
# inpath_flownet = '/home/klein/U/inference/flownet_simple_seq_9/'
# inpath_cnn = '/home/klein/U/inference/cylindrical_ref_original_seq_9/'
#
# outpath = '/home/klein/U/vis/eval_seq_9_flownet_vs_cnn/'
#
# for i in range(500,800):
#     estimation_flownet = np.array(plt.imread(inpath_flownet + str(i).zfill(3) + '_debug.png'))
#     estimation_cnn = np.array(plt.imread(inpath_cnn + str(i).zfill(3) + '_debug.png'))
#     gt = np.array(plt.imread(inpath_gt +'debug_'+ str(i).zfill(6) + '.png'))
#
#     estimation_flownet = cv2.resize(estimation_flownet, (1024,64))
#     estimation_cnn = cv2.resize(estimation_cnn, (1024,64))
#     gt = cv2.resize(gt, (1024,64))
#
#     con = np.concatenate((gt,estimation_cnn,estimation_flownet),axis=0)
#
#     # con = cv2.putText(con, "Ground Truth", (8,12), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0,1))
#     # con = cv2.putText(con, "CNN Estimation", (8, 12+45), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0, 1))
#
#     plt.imsave(outpath + str(i).zfill(3) + '_combination.png', con)
#
#     print(i)