import os
import random

import numpy as np
import tensorflow as tf

from .augment import random_crop


def resize_input(t, height, width, resized_h, resized_w):
    # Undo old resizing and apply bilinear
    t = tf.reshape(t, [resized_h, resized_w, 3])
    t = tf.expand_dims(tf.image.resize_image_with_crop_or_pad(t, height, width), 0)
    return tf.image.resize_bilinear(t, [resized_h, resized_w])


def resize_output_crop(t, height, width, channels):
    _, oldh, oldw, c = tf.unstack(tf.shape(t))
    t = tf.reshape(t, [oldh, oldw, c])
    t = tf.image.resize_image_with_crop_or_pad(t, height, width)
    return tf.reshape(t, [1, height, width, channels])


def resize_output(t, height, width, channels):
    return tf.image.resize_bilinear(t, [height, width])


def resize_output_flow(t, height, width, channels):
    batch, old_height, old_width, _ = tf.unstack(tf.shape(t), num=4)
    t = tf.image.resize_bilinear(t, [height, width])
    u, v = tf.unstack(t, axis=3)
    u *= tf.cast(width, tf.float32) / tf.cast(old_width, tf.float32)
    v *= tf.cast(height, tf.float32) / tf.cast(old_height, tf.float32)
    return tf.reshape(tf.stack([u, v], axis=3), [batch, height, width, 2])


def frame_name_to_num(name):
    stripped = name.split('.')[0].lstrip('0')
    if stripped == '':
        return 0
    return int(stripped)


class Input():
    mean = [104.920005, 110.1753, 114.785955]
    stddev = 1 / 0.0039216

    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        assert len(dims) == 2
        self.data = data
        self.dims = dims
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.normalize = normalize
        self.skipped_frames = skipped_frames

    def _resize_crop_or_pad(self, tensor):
        height, width = self.dims
        # return tf.image.resize_bilinear(tf.expand_dims(tensor, 0), [height, width])
        return tf.image.resize_image_with_crop_or_pad(tensor, height, width)

    def _resize_image_fixed(self, image):
        height, width = self.dims
        return tf.reshape(self._resize_crop_or_pad(image), [height, width, 3])

    def _normalize_image(self, image):
        return (image - self.mean) / self.stddev

    def _preprocess_image(self, image):
        image = self._resize_image_fixed(image)
        if self.normalize:
            image = self._normalize_image(image)
        return image

    def _input_images(self, image_dir, hold_out_inv=None):
        """Assumes that paired images are next to each other after ordering the
        files.
        """
        image_dir = os.path.join(self.data.current_dir, image_dir)

        filenames_1 = []
        filenames_2 = []
        image_files = os.listdir(image_dir)
        image_files.sort()

        self.number_image_files = len(image_files)
        print("Number of images: {}".format(len(image_files)))

        for i in range(len(image_files) - 1):
            filenames_1.append(os.path.join(image_dir, image_files[i]))
            filenames_2.append(os.path.join(image_dir, image_files[i + 1]))

        if hold_out_inv is not None:
            filenames = list(zip(filenames_1, filenames_2))
            random.seed(0)
            random.shuffle(filenames)
            filenames = filenames[:hold_out_inv]

            filenames_1, filenames_2 = zip(*filenames)
            filenames_1 = list(filenames_1)
            filenames_2 = list(filenames_2)

        input_1 = read_png_image(filenames_1, 1)
        input_2 = read_png_image(filenames_2, 1)
        image_1 = self._preprocess_image(input_1)
        image_2 = self._preprocess_image(input_2)
        return tf.shape(input_1), image_1, image_2

    def get_normalization(self):
        return self.mean, self.stddev

def read_png_image(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for images."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames))
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_uint8 = tf.image.decode_png(value, channels=3)
    image = tf.cast(image_uint8, tf.float32)
    return image