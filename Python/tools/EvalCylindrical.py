import numpy as np
from os import sys
sys.path.append('/home/klein/U/Masterarbeit/Python')
from math import sin, cos, radians, degrees
from tools.NumpyHelpers import XYAngleToMatrix2, similarity_transform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import svgwrite


def main(args):
    groundtruth_file = '/home/klein/U/xyzdFlow/train/kitti09/features.csv'
    in_folder = '/home/klein/U/inference/cylindrical_morefilters_seq_9/'

    np.set_printoptions(suppress=True)

    with open(groundtruth_file, 'r') as f:
        data_array = [[x if x.startswith('/') else float(x) for x in line.strip().split(',')] for line in f]

    angles = []
    angles_gt = []
    dx = []
    dx_gt = []
    dy = []
    dy_gt = []
    epes = []

    lastposition = np.identity(3)
    lastposition_gt = np.identity(3)
    dwg = svgwrite.Drawing(filename='test.svg', size=(2200, 2000))
    dwg.add(dwg.text('Ground Truth', insert=(40, 80), style='font-size:40', stroke='red', fill='red'))
    dwg.add(dwg.text('Estimation', insert=(500, 80), style='font-size:40', stroke='blue', fill='blue'))
    lines_gt = dwg.add(dwg.g(id='lines', stroke='red', stroke_width=2, transform='translate(300,600)'))
    lines_est = dwg.add(dwg.g(id='lines', stroke='blue', stroke_width=2, transform='translate(300,600)'))

    for i, data_line in enumerate(data_array[:600]):
        #if i < 90: continue

        fieldX = np.load(in_folder + str(i).zfill(3) + '_estimation_dX.npy')
        fieldY = np.load(in_folder + str(i).zfill(3) + '_estimation_dY.npy')

        image = np.load(data_line[0][:-3]+'npy')
        layerX = image[:, :, 0]
        layerY = image[:, :, 1]

        image_gt = np.load(data_line[2][:-3]+'npy')
        fieldXgt = image_gt[:, :, 0]
        fieldYgt = image_gt[:, :, 1]

        epe = np.nanmean(np.where(np.abs(layerX) > 0.001, np.sqrt(np.square(fieldX-fieldXgt) + np.square(fieldY-fieldYgt)), np.nan))

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

        points1X = np.where(np.abs(layerX) > 0.001, layerX, np.nan)
        points1Y = np.where(np.abs(layerX) > 0.001, layerY, np.nan)
        points2X = np.where(np.abs(layerX) > 0.001, layerX + fieldX, np.nan)
        points2Y = np.where(np.abs(layerX) > 0.001, layerY + fieldY, np.nan)

        points1X = points1X[~np.isnan(points1X)]
        points1Y = points1Y[~np.isnan(points1Y)]
        points2X = points2X[~np.isnan(points2X)]
        points2Y = points2Y[~np.isnan(points2Y)]

        points1 = np.stack((points1X, points1Y),axis=1)
        points2 = np.stack((points2X, points2Y), axis=1)

        R, t = similarity_transform(points1, points2)

        dx_gt.append(data_array[i][-3])
        dx.append(t[0]*10)
        dy_gt.append(data_array[i][-2])
        dy.append(t[1]*10)
        angles.append(-np.arcsin(R[1, 0]))
        angles_gt.append(np.radians(data_array[i][-1]))





        # svg visualization
        est_x = t[0]*10
        est_y = t[1]*10
        est_phi = degrees(-np.arcsin(R[1, 0]))

        estimation = np.array([-est_x, -est_y, est_phi])
        gt = np.array(data_line[-3:])

        tf_estimation = XYAngleToMatrix2(estimation)
        tf_gt = XYAngleToMatrix2(gt)

        newposition = lastposition @ tf_estimation
        newposition_gt = lastposition_gt @ tf_gt

        f = 0.5

        # to svg:
        lines_gt.add(dwg.line(start=(lastposition_gt[0, 2] * f, lastposition_gt[1, 2] * f),
                              end=(newposition_gt[0, 2] * f, newposition_gt[1, 2] * f)))
        lines_est.add(dwg.line(start=(lastposition[0, 2] * f, lastposition[1, 2] * f),
                               end=(newposition[0, 2] * f, newposition[1, 2] * f)))

        lastposition = newposition
        lastposition_gt = newposition_gt

    #
    dwg.save()

    plt.subplot(311)
    plt.legend(handles=[mpatches.Patch(color='C1', label='Ground Truth'), mpatches.Patch(color='C0', label='Estimation')])
    plt.title('X-Bewegung')
    plt.plot(range(len(dx)), dx, range(len(dx_gt)), -np.array(dx_gt))
    plt.subplot(312)
    plt.legend(handles=[mpatches.Patch(color='C1', label='Ground Truth'), mpatches.Patch(color='C0', label='Estimation')])
    plt.title('Y-Bewegung')
    plt.plot(range(len(dy)), dy, range(len(dy_gt)), -np.array(dy_gt))
    plt.subplot(313)
    plt.legend(handles=[mpatches.Patch(color='C1', label='Ground Truth'), mpatches.Patch(color='C0', label='Estimation')])
    plt.title('Winkel')
    plt.plot(range(len(angles)), angles, range(len(angles_gt)), np.array(angles_gt))
    plt.show()


    print("Final diff:")
    print(np.nanmean(np.array(epes)))


if __name__ == '__main__':
    main(sys.argv)