from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf

s = tf.Session()

mypath = '/home/klein/U/depthimageFlow/train/kitti08/'

llist = listdir(mypath)

llist.sort()

for f in llist:
    f = join(mypath, f)
    if isfile(f) and '16bit' in f:

        aufl = 200
        if 'flow' in f: aufl = 2000

        # convert:
        image_str = tf.read_file(f)
        image = (tf.cast(tf.image.decode_png(image_str, channels=3, dtype=tf.uint16), dtype=tf.float32) - 32768) / aufl

        arr = np.array(s.run(image))

        np.save(f[:-3] + 'npy', arr)

        print(f)