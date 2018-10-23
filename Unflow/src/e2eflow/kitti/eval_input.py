import os
import sys

import numpy as np
import tensorflow as tf
import random

from ..core.eval_input import read_png_image, Input

class KITTIInput(Input):
    def __init__(self, data, batch_size, dims, layers, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, layers, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def _input_train(self, image_dir, hold_out_inv=None):
        input_shape, im1, im2 = self._input_images(image_dir, hold_out_inv)
        return tf.train.batch(
            [im1, im2, input_shape],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def input_grid_map(self, hold_out_inv=None):
        print("start inputing grid map")
        evaluate_dir = os.path.join(self.data.current_dir, 'evaluate')
        image_dir = os.path.join(evaluate_dir, os.listdir(evaluate_dir)[0])
        return self._input_train(image_dir,
                                 hold_out_inv)
