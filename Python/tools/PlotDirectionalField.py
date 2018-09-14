from matplotlib.colors import hsv_to_rgb
import matplotlib
import matplotlib.cm
from math import pi
import numpy as np
import tensorflow as tf
#import skimage.measure


def directionalFieldTF(layerX, layerY, scale=0):
    saturation = tf.sqrt(tf.square(layerX) + tf.square(layerY))
    if scale == 0:
        amax = tf.reduce_max(tf.abs(saturation))
        saturation = saturation/amax
    else:
        saturation = saturation/scale
        saturation = tf.where(saturation > 1, tf.ones_like(saturation), tf.ones_like(saturation)*saturation)
    hue = (tf.atan2(layerY, layerX)+pi)/2/pi
    value = tf.ones(hue.shape, dtype=tf.float32)

    return tf.image.hsv_to_rgb(tf.stack([hue, saturation, value], axis=2))


def directionalField(layerX, layerY, scale=0):
    saturation = np.sqrt(np.square(layerX)+np.square(layerY))
    if scale == 0:
        amax = np.amax(np.abs(saturation))
        saturation = saturation/amax
    else:
        saturation = saturation/scale
        saturation = np.where(saturation > 1, 1, saturation)
    hue = (np.arctan2(layerY, layerX)+pi)/2/pi
    value = np.ones(hue.shape, dtype=np.float32)

    return hsv_to_rgb(np.stack([hue, saturation, value], axis=2))


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.

    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.

    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')

    Example usage:

    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```

    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = tf.constant([cm(c) for c in range(255)], dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value



def main():
    import cv2
    import matplotlib.pyplot as plt


    # resolution = 1000
    # test = np.tile(np.arange(5,-5,-10/resolution),(resolution,1))
    # img = directionalField(test.transpose(), test)
    # plt.imshow(img)
    # plt.show()


    # with open('/home/klein/U/gridmapsFlow/train/kitti00/features.csv', 'r') as f:
    #     data = [line.strip() for line in f]
    #
    # for i,d in enumerate(data):
    #     d = d.split(',')
    #     layerX_file = d[2]
    #     layerY_file = d[3]
    #     # layerX_file = '/home/klein/U/gridmapsFlow/train/kitti00/0_euclidean_vX_76507950.png'
    #     # layerY_file = '/home/klein/U/gridmapsFlow/train/kitti00/0_euclidean_vY_76507950.png'
    #
    #     layerX = np.array(cv2.imread(layerX_file, cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128
    #     layerY = np.array(cv2.imread(layerY_file, cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128
    #
    #     img = directionalField(layerX, layerY)
    #
    #     f = 2.5
    #     for x in np.arange(50, 1200, 50):
    #         for y in np.arange(50, 1200, 50):
    #             img = cv2.line(img, (y, x), (y - int(layerY[x,y]*f), x - int(layerX[x,y]*f)), 0)
    #             img = cv2.circle(img, (y,x), 4, 0)
    #
    #     plt.imsave('/home/klein/U/tmp/img'+str(i).zfill(3)+'.png',img)
    #     print(i+1,'of',len(data))
    #     # plt.imshow(img)
    #     # plt.show()


    with open('/home/klein/U/gridmapsFlow/eval/kitti00/features.csv', 'r') as f:
        data = [line.strip() for line in f]

    for i,d in enumerate(data):
        d = d.split(',')
        layerX_file = d[2]
        layerY_file = d[3]

        layerX = np.array(cv2.imread(layerX_file, cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128
        layerY = np.array(cv2.imread(layerY_file, cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128

        img = directionalField(layerX, layerY)

        # f = 2.5
        # for x in np.arange(50, 1200, 50):
        #     for y in np.arange(50, 1200, 50):
        #         img = cv2.line(img, (y, x), (y - int(layerY[x,y]*f), x - int(layerX[x,y]*f)), 0)
        #         img = cv2.circle(img, (y,x), 4, 0)

        plt.imsave('/home/klein/U/tmp/img'+str(i).zfill(3)+'.png',img)
        print(i+1,'of',len(data))
        # plt.imshow(img)
        # plt.show()

    # in_folder = '/home/klein/U/tmp_eval'
    # for i in range(1000):
    #     layerX = np.load(in_folder + '/' + str(i).zfill(3) + '_estimation_dX.npy')
    #     layerY = np.load(in_folder + '/' + str(i).zfill(3) + '_estimation_dY.npy')
    #
    #     img = directionalField(layerX, layerY)
    #
    #     gtX = (np.array(cv2.imread('', cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128)[300:900,300:900]
    #     #gtX = skimage.measure.block_reduce(gtX, (2,2), np.max)
    #     gtY = (np.array(cv2.imread('', cv2.IMREAD_GRAYSCALE), dtype=np.float32) - 128)[300:900,300:900]
    #     #gtY = skimage.measure.block_reduce(gtY, (2, 2), np.max)
    #
    #
    #     plt.imsave('/home/klein/U/tmp/img'+str(i).zfill(3)+'.png',img)
    #     print(i+1,'of',1000)


if __name__ == "__main__":
    main()
