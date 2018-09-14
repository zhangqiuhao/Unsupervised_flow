import tensorflow as tf
import numpy as np

class SaveTrainableParamsCount(tf.train.SessionRunHook):
    """Hook which saves total count of trainable parameters.
    logdir is intended to be the same path as passed to the estimator."""

    def __init__(self, logdir):
        super().__init__()
        self._logdir = logdir

    def begin(self):
        tvars = tf.trainable_variables()
        count = np.sum([np.prod(var.get_shape().as_list()) for var in tvars])
        with open('%s/params_%i.txt' % (self._logdir, int(count)), mode='w') as txt_file:
            txt_file.write('This network contains %i trainable parameters.' % int(count))
