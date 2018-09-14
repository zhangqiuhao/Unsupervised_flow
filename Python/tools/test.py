import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

s = tf.Session()

folder = '/home/klein/U/gridmapsFlow/eval/kitti08/'

for i in range(5000):
    image_str_gt = tf.read_file(folder + 'flow_16bit_'+str(i).zfill(6)+'_'+str(i+1).zfill(6)+'.png')
    image_gt = (tf.cast(tf.image.decode_png(image_str_gt, channels=3, dtype=tf.uint16), dtype=tf.float32) - 32768) / 1000
    image_gt.set_shape([600,600,3])

    image_gt = image_gt * 10

    img = s.run(image_gt)

    img = np.array(img, dtype=np.float32)

    np.save(folder + 'flow_16bit_'+str(i).zfill(6)+'_'+str(i+1).zfill(6), img)

    print(i)