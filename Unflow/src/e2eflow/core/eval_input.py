import os
import random
import png

import numpy as np
import tensorflow as tf

from .augment import random_crop


def resize_input(t, height, width, layers, resized_h, resized_w):
    # Undo old resizing and apply bilinear
    t = tf.reshape(t, [resized_h, resized_w, layers])
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
    stddev = 1 / 0.0039216
    mean = []

    def __init__(self, data, batch_size, dims, layers, num_layers, mask_layers, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        assert len(dims) == 2
        self.data = data
        self.dims = dims
        self.layers = layers
        self.mask_layers = mask_layers
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.normalize = normalize
        self.skipped_frames = skipped_frames
        self.num_layers = num_layers

    def _resize_crop_or_pad(self, tensor):
        height, width = self.dims
        # return tf.image.resize_bilinear(tf.expand_dims(tensor, 0), [height, width])
        return tf.image.resize_image_with_crop_or_pad(tensor, height, width)

    def _resize_image_fixed(self, image, layer):
        height, width = self.dims
        return tf.reshape(self._resize_crop_or_pad(image), [height, width, layer])

    def _normalize_image(self, image):
        return (image - self.mean) / self.stddev

    def _preprocess_image(self, image, layer):
        image = self._resize_image_fixed(image, layer)
        if self.normalize:
            image = self._normalize_image(image)
        return image

    def _input_images(self, current_dir, hold_out_inv=None):
        """Assumes that paired images are next to each other after ordering the
        files.
        """
        calib_dir = os.path.join(current_dir, 'calib')
        image_dir = os.path.join(current_dir, 'gridmap')

        filenames_1 = []
        filenames_2 = []
        calib_files = os.listdir(calib_dir)
        calib_files.sort()

        for i in range(len(calib_files) - 1):
            filenames_1.append(os.path.join(image_dir, calib_files[i].split('.')[0]))
            filenames_2.append(os.path.join(image_dir, calib_files[i + 1].split('.')[0]))

        if hold_out_inv is not None:
            filenames = list(zip(filenames_1, filenames_2))
            random.seed(0)
            random.shuffle(filenames)
            filenames = filenames[:hold_out_inv]

            filenames_1, filenames_2 = zip(*filenames)
            filenames_1 = list(filenames_1)
            filenames_2 = list(filenames_2)

        input_1, mask_image_1 = read_png_image(filenames_1, self.layers, self.mask_layers, 1)
        input_2, mask_image_2 = read_png_image(filenames_2, self.layers, self.mask_layers, 1)

        input_1 = tf.concat([x for x in input_1], axis=-1)
        input_2 = tf.concat([x for x in input_2], axis=-1)

        height, width = self.dims
        if mask_image_1 is None or mask_image_2 is None:
            mask_image_1 = tf.ones([height, width], tf.float32)
            mask_image_2 = tf.ones([height, width], tf.float32)
        else:
            mask_image_1 = self._preprocess_image(mask_image_1, len(self.mask_layers))
            mask_image_2 = self._preprocess_image(mask_image_2, len(self.mask_layers))

        image_1 = self._preprocess_image(input_1, self.num_layers)
        image_2 = self._preprocess_image(input_2, self.num_layers)

        return tf.shape(input_1), image_1, image_2, mask_image_1, mask_image_2

    def get_normalization(self):
        #[104.920005, 110.1753, 114.785955]
        self.mean = [127.5] * self.num_layers
        return self.mean, self.stddev


def read_png_image(filenames, layers, mask_layers, num_epochs=None):
    """Given a list of filenames, constructs a reader op for images."""
    image = []
    reader = tf.WholeFileReader()
    if layers is not None:
        for _, layer in enumerate(layers):
            layer = '_' + layer + '.png'
            filenames_with_layer = [i+layer for i in filenames]
            filename_queue = tf.train.string_input_producer(filenames_with_layer,
                                                            shuffle=False, capacity=len(filenames_with_layer))
            _, value = reader.read(filename_queue)
            image_uint8 = tf.image.decode_png(value, channels=1)
            image_float32 = tf.cast(image_uint8, tf.float32)
            image.append(image_float32)
    if mask_layers is not None and mask_layers in ['z_max_occlusions_cartesian']:
        layer = '_' + mask_layers[0] + '.png'
        filenames_with_layer = [i + layer for i in filenames]
        filename_queue = tf.train.string_input_producer(filenames_with_layer,
                                                        shuffle=False, capacity=len(filenames_with_layer))
        _, value = reader.read(filename_queue)
        image_uint8 = tf.image.decode_png(value, channels=1)
        mask_image_float32 = tf.cast(image_uint8, tf.float32)

        if mask_layers == 'z_max_occlusions_cartesian':
            mask_image_float32 = 1.0 - mask_image_float32 / 255.0
    else:
        mask_image_float32 = None
    return image, mask_image_float32
