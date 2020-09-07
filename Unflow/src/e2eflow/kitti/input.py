import os
import sys

import numpy as np
import tensorflow as tf
import random

from ..core.input import read_png_image, Input
from ..core.augment import random_crop


class KITTIInput(Input):
    def __init__(self, data, batch_size, dims, layers, num_layers, mask_layers, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, layers, num_layers, mask_layers, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def _input_images(self, image_dir, hold_out_inv=None):
        """Assumes that paired images are next to each other after ordering the
        files.
        """
        image_dir = os.path.join(self.data.current_dir, image_dir)

        filenames_1 = []
        filenames_2 = []

        for i in range(16):
            filenames_1.append(os.path.join(image_dir, str(2*i).zfill(6)))
            filenames_2.append(os.path.join(image_dir, str(2*i+1).zfill(6)))

        if hold_out_inv is not None:
            filenames = list(zip(filenames_1, filenames_2))
            random.seed(0)
            random.shuffle(filenames)
            filenames = filenames[:hold_out_inv]

            filenames_1, filenames_2 = zip(*filenames)
            filenames_1 = list(filenames_1)
            filenames_2 = list(filenames_2)

        input_1 = read_png_image(filenames_1, self.layers, self.mask_layers, 1)
        input_2 = read_png_image(filenames_2, self.layers, self.mask_layers, 1)
        image_1 = self._preprocess_image(input_1, self.num_layers)
        image_2 = self._preprocess_image(input_2, self.num_layers)
        return tf.shape(image_1), image_1, image_2

    def _input_train(self, image_dir, hold_out_inv=None):
        input_shape, im1, im2 = self._input_images(image_dir, hold_out_inv)
        return tf.train.batch(
            [im1, im2, input_shape],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def input_validation(self, hold_out_inv=None):
        print("start inputing validation")
        return self._input_train('validation',
                                 hold_out_inv)

