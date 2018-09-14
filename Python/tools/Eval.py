import numpy as np
import svgwrite
from os import walk, path, sys
sys.path.append('/home/klein/U/Masterarbeit/Python')
from math import sin, cos, radians
from tools.NumpyHelpers import XYAngleToMatrix2, similarity_transform
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi, degrees
import skimage.measure
from tools.PlotDirectionalField import directionalField
from matplotlib.colors import LinearSegmentedColormap


def main(args):
    groundtruth_file = '/home/klein/U/gridmapsfull/train/kitti09/gridmaps.csv'
    in_folder = '/home/klein/U/inference/gridmaps_seq_9'
    number_of_run = 0


    #########
    start = 500
    end = 800
    ###########

    np.set_printoptions(suppress=True)

    lastposition = np.identity(3)
    lastposition_gt = np.identity(3)
    dwg = svgwrite.Drawing(filename='test.svg', size=(2200, 2000))
    dwg.add(dwg.text('Ground Truth', insert=(40, 80), style='font-size:40', stroke='red', fill='red'))
    dwg.add(dwg.text('Estimation', insert=(500, 80), style='font-size:40', stroke='blue', fill='blue'))
    lines_gt = dwg.add(dwg.g(id='lines', stroke='red', stroke_width=2, transform='translate(300,600)'))
    lines_est = dwg.add(dwg.g(id='lines', stroke='blue', stroke_width=2, transform='translate(300,600)'))

    with open(groundtruth_file, 'r') as f:
        data_array = [[x if x.startswith('/') else float(x) for x in line.strip().split(',')] for line in f]

    with open('/home/klein/U/gridmapsFlow_old/eval/kitti09/features.csv', 'r') as f:
        data_array2 = [[x.replace('gridmapsFlow/','gridmapsFlow_old/') if x.startswith('/') else float(x) for x in line.strip().split(',')] for line in f]

    posY = np.tile(np.arange(300, -300, -2, dtype=np.float32), (300, 1)) + 0.00001
    posX = posY.transpose()

    angles = []
    angles_gt = []

    dx = []
    dx_gt = []
    dy = []
    dy_gt = []
    epes = []


    for i, data_line in enumerate(data_array[:end]):
        if i < start: continue

        fieldX = np.load(in_folder + '/' + str(i).zfill(4) + '_estimation_dX.npy')
        fieldY = np.load(in_folder + '/' + str(i).zfill(4) + '_estimation_dY.npy')

        # max-min-pool:
        signmask = np.kron(np.sign(skimage.measure.block_reduce(fieldX, (4, 4), np.mean)), np.ones((4,4)))
        # fieldX = skimage.measure.block_reduce(fieldX * signmask, (4,4), np.max)
        # fieldX = np.kron(fieldX, np.ones((4,4))) * signmask
        # signmask = np.kron(np.sign(skimage.measure.block_reduce(fieldY, (4, 4), np.mean)), np.ones((4,4)))
        # fieldY = skimage.measure.block_reduce(fieldY * signmask, (4,4), np.max)
        # fieldY = np.kron(fieldY, np.ones((4,4))) * signmask

        layer1 = (np.array(cv2.imread(data_array2[i][0], cv2.IMREAD_GRAYSCALE)))[300:900,300:900]
        #layer1 = skimage.measure.block_reduce(layer1, (2,2), np.max)
        layer1 = cv2.resize(layer1, (300,300))

        # reducing mask to bigger blobs
        # _, layer1 = cv2.threshold(layer1, 254, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((2, 2), np.uint8)
        # layer1 = cv2.dilate(layer1,kernel,iterations = 1)

        fieldXgt = (np.array(cv2.imread(data_array2[i][2], cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128)[300:900,300:900]
        signmask = np.sign(skimage.measure.block_reduce(fieldXgt, (2, 2), np.mean))
        fieldXgt = skimage.measure.block_reduce(fieldXgt * np.kron(signmask, np.ones((2,2))), (2,2), np.max)
        fieldXgt = fieldXgt * signmask

        fieldYgt = (np.array(cv2.imread(data_array2[i][3], cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128)[300:900,300:900]
        signmask = np.sign(skimage.measure.block_reduce(fieldYgt, (2, 2), np.mean))
        fieldYgt = skimage.measure.block_reduce(fieldYgt * np.kron(signmask, np.ones((2,2))), (2,2), np.max)
        fieldYgt = fieldYgt * signmask

        epe = np.nanmean(np.where(layer1 != 255, np.sqrt(np.square(fieldX-fieldXgt) + np.square(fieldY-fieldYgt)), np.nan))

        epes.append(epe)
        print(epe)


        # fieldXt = tf.reshape(fieldX, [1,300,300,1])
        #
        # signs = tf.sign(tf.layers.average_pooling2d(fieldXt, 4, 4, 'same'))
        # signs = tf.image.resize_images(signs, [300, 300], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # fieldXt = tf.layers.max_pooling2d(fieldXt*signs, 4, 4, 'same')
        # fieldXt = tf.image.resize_images(fieldXt, [300, 300], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # fieldXt = fieldXt * signs
        #
        # fieldXt = tf.reshape(fieldXt, [300, 300])
        # fieldXt = s.run(fieldXt)
        #
        # plt.subplot(211)
        # plt.imshow(fieldXgt)
        #
        # plt.subplot(212)
        # plt.imshow(np.where(layer1 != 255, fieldXt, np.nan))
        # plt.show()
        #
        # break


        # fgt = directionalField(fieldXgt, fieldYgt)
        # plt.subplot(231)
        # plt.imshow(fgt)
        # fgt = np.where(np.stack([layer1,layer1,layer1],axis=2) != 255, fgt, np.nan)
        #
        # f = directionalField(fieldX, fieldY)
        # plt.subplot(232)
        # plt.imshow(f)
        #
        # f = np.where(np.stack([layer1,layer1,layer1],axis=2) != 255, f, np.nan)
        #
        # plt.subplot(234)
        # plt.imshow(fgt)
        # plt.subplot(235)
        # plt.imshow(f)
        # plt.subplot(236)
        #
        # plt.subplot(233)
        # ax = plt.gca()
        # ax.set_facecolor('black')
        # cm = LinearSegmentedColormap.from_list(
        #     'mylist', [(0, 1, 0), (1, 0, 0)], N=100)
        # plt.imshow(np.where(layer1 != 255, np.sqrt(np.square(fieldX-fieldXgt) + np.square(fieldY-fieldYgt)), np.nan), cmap=cm)
        # plt.show()


        # combine field estimation to one single gridmap estimation

        points1X = np.where(layer1 != 255, posX, np.nan)
        points1Y = np.where(layer1 != 255, posY, np.nan)
        points2X = np.where(layer1 != 255, posX + fieldX, np.nan)
        points2Y = np.where(layer1 != 255, posY + fieldY, np.nan)

        points1X = points1X[~np.isnan(points1X)]
        points1Y = points1Y[~np.isnan(points1Y)]
        points2X = points2X[~np.isnan(points2X)]
        points2Y = points2Y[~np.isnan(points2Y)]

        points1 = np.stack((points1X, points1Y),axis=1)
        points2 = np.stack((points2X, points2Y), axis=1)

        R, t = similarity_transform(points1, points2)

        dx_gt.append(data_array[i][-3])
        dx.append(t[0])
        dy_gt.append(data_array[i][-2])
        dy.append(t[1])
        angles.append(-np.arcsin(R[1, 0]))
        angles_gt.append(np.radians(data_array[i][-1]))





        # est_x = t[0]
        # est_y = t[1]
        # est_phi = degrees(-np.arcsin(R[1, 0]))
        #
        # estimation = np.array([-est_x, -est_y, est_phi])
        # gt = np.array(data_line[-3:])
        #
        # # visualization
        # tf_estimation = XYAngleToMatrix2(estimation)
        # tf_gt = XYAngleToMatrix2(gt)
        #
        # newposition = lastposition @ tf_estimation
        # newposition_gt = lastposition_gt @ tf_gt
        #
        # f = 0.2
        #
        # # to svg:
        # lines_gt.add(dwg.line(start=(lastposition_gt[0, 2] * f, lastposition_gt[1, 2] * f),
        #                       end=(newposition_gt[0, 2] * f, newposition_gt[1, 2] * f)))
        # lines_est.add(dwg.line(start=(lastposition[0, 2] * f, lastposition[1, 2] * f),
        #                        end=(newposition[0, 2] * f, newposition[1, 2] * f)))
        #
        # lastposition = newposition
        # lastposition_gt = newposition_gt

    dwg.save()




    angles_gt2 = []
    dx_gt2 = []
    dy_gt2 = []

    from tools.NumpyHelpers import rotationMatrixToEulerAngles
    inpath_tf_oxt = '/mrtstorage/datasets/kitti/odometry/all_py/poses/' + str(9).zfill(2) + '.txt'

    inpath_calib = '/mrtstorage/datasets/kitti/odometry/data_odometry_calib/sequences/' + str(9).zfill(2) + '/calib.txt'
    with open(inpath_calib, 'r') as f:
        data_calib = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(': ')[1].split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    tf_cam_to_velo = data_calib[4]
    tf_velo_to_cam = np.linalg.inv(data_calib[4])

    with open(inpath_tf_oxt, 'r') as f:
        oxt = [np.matrix(np.concatenate((np.reshape([float(x) for x in line.strip().split(' ')], (3, 4)), [[0, 0, 0, 1]]))) for line in f]

    for i, tf_this_frame in enumerate(oxt[:end-1]):
        if i < start: continue
        tf_next_frame = oxt[i + 1]
        tf_between_frames = tf_velo_to_cam @ np.linalg.inv(tf_this_frame) @ tf_next_frame @ tf_cam_to_velo

        dx_gt2.append(-tf_between_frames[0,3])
        dy_gt2.append(-tf_between_frames[1,3])
        angles_gt2.append(np.degrees(rotationMatrixToEulerAngles(tf_between_frames[0:4,0:4])[2]))





    #
    # plt.subplot(311)
    # plt.legend(handles=[mpatches.Patch(color='C0', label='Ground Truth'), mpatches.Patch(color='C1', label='Estimation')])
    # plt.title('X-Bewegung')
    # plt.plot(range(len(dx_gt2)), -np.array(dx_gt2), range(len(dx)), -np.array(dx)/10)
    # plt.subplot(312)
    # plt.legend(handles=[mpatches.Patch(color='C0', label='Ground Truth'), mpatches.Patch(color='C1', label='Estimation')])
    # plt.title('Y-Bewegung')
    # plt.plot(range(len(dy_gt2)), -np.array(dy_gt2), range(len(dy)), -np.array(dy)/10)
    # plt.subplot(313)
    # plt.legend(handles=[mpatches.Patch(color='C0', label='Ground Truth'), mpatches.Patch(color='C1', label='Estimation')])
    # plt.title('Winkel')
    # plt.plot(range(len(angles_gt2)), np.array(angles_gt2), range(len(angles)), np.degrees(np.array(angles)))
    # plt.show()


    c1 = '#003d84'
    c2 = '#c50001'

    plt.subplot(311)
    plt.grid(b=True, c='#cccccc')
    plt.legend(handles=[mpatches.Patch(color=c1, label='Ground Truth'), mpatches.Patch(color=c2, label='Estimation')])
    #plt.title('X-Bewegung')
    plt.plot(range(len(dx_gt2)), -np.array(dx_gt2), c1, range(len(dx)), -np.array(dx)/10, c2)
    plt.subplot(312)
    plt.grid(b=True, c='#cccccc')
    plt.legend(handles=[mpatches.Patch(color=c1, label='Ground Truth'), mpatches.Patch(color=c2, label='Estimation')])
    #plt.title('Y-Bewegung')
    plt.plot(range(len(dy_gt2)), -np.array(dy_gt2), c1, range(len(dy)), -np.array(dy)/10, c2)
    plt.subplot(313)
    plt.grid(b=True, c='#cccccc')
    plt.legend(handles=[mpatches.Patch(color=c1, label='Ground Truth'), mpatches.Patch(color=c2, label='Estimation')])
    #plt.title('Winkel')
    plt.plot(range(len(angles_gt2)), np.array(angles_gt2), c1, range(len(angles)), np.degrees(np.array(angles)), c2)
    plt.show()

    print("Final diff:")
    print(np.nanmean(np.array(epes)))

    print("Average X error:")
    print(np.abs(dx-(-np.array(dx_gt))).mean())

    print("Average Y error:")
    print(np.abs(dy-(-np.array(dy_gt))).mean())

    print("Average Angle error:")
    print(np.abs(angles-(np.array(angles_gt))).mean())

if __name__ == '__main__':
    main(sys.argv)