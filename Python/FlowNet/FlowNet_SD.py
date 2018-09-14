import tensorflow as tf
from tools.PlotDirectionalField import directionalFieldTF
from tensorflow.python.ops.init_ops import Initializer
import numpy as np


class LoadInitializer(Initializer):

    def __init__(self, dtype=tf.float32, name='', trainable=True):
        self.dtype = tf.as_dtype(dtype)
        self.filename = '/home/klein/U/extracted_weights/flownet_sd/' + name + '.npy'

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        print('Loading', self.filename)
        data = np.load(self.filename)
        tensor = tf.reshape(tf.convert_to_tensor(data, dtype=dtype), shape)

        return tensor

    def get_config(self):
        return {"dtype": self.dtype.name}


def conv(a, filters, name, kernel_size=1, strides=1, activation=tf.nn.leaky_relu, fr=False, reuse=None, trainable=True):
    if fr:
        return tf.layers.conv2d(a, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name=name, activation=activation, reuse=reuse, kernel_initializer=LoadInitializer(name=name+'_weights', trainable=trainable),  bias_initializer=LoadInitializer(name=name+'_biases', trainable=trainable), trainable=trainable)
    else:
        return tf.layers.conv2d(a, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name=name, activation=activation, reuse=reuse, trainable=trainable)


def deconv(a, filters, name=None, kernel_size=1, strides=1, activation=tf.nn.leaky_relu, fr=False, trainable=True):
    if fr:
        return tf.layers.conv2d_transpose(a, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name=name, activation=activation, kernel_initializer=LoadInitializer(name=name+'_weights', trainable=trainable),  bias_initializer=LoadInitializer(name=name+'_biases', trainable=trainable), trainable=trainable)
    else:
        return tf.layers.conv2d_transpose(a, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name=name, activation=activation, trainable=trainable)


def model_fn(features, labels, mode, params):
    firstrun = params['FIRSTRUN']

    # resizing:
    features = tf.image.resize_images(features, [64, 1024])
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.image.resize_images(labels, [64, 1024])

    x = features

    x = conv(x, filters=64, kernel_size=3, fr=firstrun, name='conv0')
    x = conv(x, filters=64, kernel_size=3, strides=2, fr=firstrun, name='conv1')
    x = conv(x, filters=128, kernel_size=3, fr=firstrun, name='conv1_1')
    x = conv(x, filters=128, kernel_size=3, strides=2, fr=firstrun, name='conv2')
    x_conv2 = x
    x = conv(x, filters=128, kernel_size=3, fr=firstrun, name='conv2_1')
    x = conv(x, filters=256, kernel_size=3, strides=2, fr=firstrun, name='conv3')
    x = conv(x, filters=256, kernel_size=3, fr=firstrun, name='conv3_1')
    x_conv3_1 = x
    x = conv(x, filters=512, kernel_size=3, strides=2, fr=firstrun, name='conv4')
    x = conv(x, filters=512, kernel_size=3, fr=firstrun, name='conv4_1')
    x_conv4_1 = x
    x = conv(x, filters=512, kernel_size=3, strides=2, fr=firstrun, name='conv5')
    x = conv(x, filters=512, kernel_size=3, fr=firstrun, name='conv5_1')
    x_conv5_1 = x
    x = conv(x, filters=1024, kernel_size=3, strides=2, fr=firstrun, name='conv6')
    x = conv(x, filters=1024, kernel_size=3, fr=firstrun, name='conv6_1')

    # refinement
    predict_6 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, name='predict_flow6')
    x = deconv(x, filters=512, kernel_size=4, strides=2, fr=firstrun, name='deconv5')
    upsample_6to5 = deconv(predict_6, filters=2, kernel_size=4, strides=2, activation=None, fr=firstrun, name='upsample_flow6to5')
    x = tf.concat([x_conv5_1, x, upsample_6to5], axis=3, name='concat_1')
    inter = conv(x, filters=512, kernel_size=3, activation=None, fr=firstrun, name='interconv5')

    predict_5 = conv(inter, filters=2, kernel_size=3, activation=None, fr=firstrun, name='predict_flow5')
    x = deconv(x, filters=256, kernel_size=4, strides=2, fr=firstrun, name='deconv4')
    upsample_5to4 = deconv(predict_5, filters=2, kernel_size=4, strides=2, activation=None, fr=firstrun, name='upsample_flow5to4')
    x = tf.concat([x_conv4_1, x, upsample_5to4], axis=3, name='concat_2')
    inter = conv(x, filters=256, kernel_size=3, activation=None, fr=firstrun, name='interconv4')

    predict_4 = conv(inter, filters=2, kernel_size=3, activation=None, fr=firstrun, name='predict_flow4')
    x = deconv(x, filters=128, kernel_size=4, strides=2, fr=firstrun, name='deconv3')
    upsample_4to3 = deconv(predict_4, filters=2, kernel_size=4, strides=2, activation=None, fr=firstrun, name='upsample_flow4to3')
    x = tf.concat([x_conv3_1, x, upsample_4to3], axis=3, name='concat_3')
    inter = conv(x, filters=128, kernel_size=3, activation=None, fr=firstrun, name='interconv3')

    predict_3 = conv(inter, filters=2, kernel_size=3, activation=None, fr=firstrun, name='predict_flow3')
    x = deconv(x, filters=64, kernel_size=4, strides=2, fr=firstrun, name='deconv2')
    upsample_3to2 = deconv(predict_3, filters=2, kernel_size=4, strides=2, activation=None, fr=firstrun, name='upsample_flow3to2')
    x = tf.concat([x_conv2, x, upsample_3to2], axis=3, name='concat_4')
    x = conv(x, filters=64, kernel_size=3, activation=None, fr=firstrun, name='interconv2')

    predict_2 = conv(x, filters=3, kernel_size=3, activation=None, fr=False, name='predict_2')
    x = tf.image.resize_bilinear(predict_2, [64, 1024], align_corners=True)



    # for prediction only:
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"predictions": x})

    if params['LOSS'] == 'special':
        losses = []

        # L2 loss between predict_flow6, blob23 (weighted w/ 0.32)
        size = [predict_6.shape[1], predict_6.shape[2]]
        downsampled_flow6 = tf.image.resize_bilinear(labels, size)
        losses.append(tf.losses.mean_squared_error(downsampled_flow6, predict_6))

        # L2 loss between predict_flow5, blob28 (weighted w/ 0.08)
        size = [predict_5.shape[1], predict_5.shape[2]]
        downsampled_flow5 = tf.image.resize_bilinear(labels, size)
        losses.append(tf.losses.mean_squared_error(downsampled_flow5, predict_5))

        # L2 loss between predict_flow4, blob33 (weighted w/ 0.02)
        size = [predict_4.shape[1], predict_4.shape[2]]
        downsampled_flow4 = tf.image.resize_bilinear(labels, size)
        losses.append(tf.losses.mean_squared_error(downsampled_flow4, predict_4))

        # L2 loss between predict_flow3, blob38 (weighted w/ 0.01)
        size = [predict_3.shape[1], predict_3.shape[2]]
        downsampled_flow3 = tf.image.resize_bilinear(labels, size)
        losses.append(tf.losses.mean_squared_error(downsampled_flow3, predict_3))

        # L2 loss between predict_flow2, blob43 (weighted w/ 0.005)
        size = [predict_2.shape[1], predict_2.shape[2]]
        downsampled_flow2 = tf.image.resize_bilinear(labels, size)
        losses.append(tf.losses.mean_squared_error(downsampled_flow2, predict_2))

        loss = tf.losses.compute_weighted_loss(losses, [0.32, 0.08, 0.02, 0.01, 0.005])
    else :
        # calculate loss:
        mask = tf.stack([features[:, :, :, 0], features[:, :, :, 0], features[:, :, :, 0]], axis=3)
        weight_matrix = tf.where(tf.abs(mask) < 0.001, tf.ones_like(mask), tf.ones_like(mask)*params['LOSS_WEIGHT_OCCUPIED'])

        loss = tf.losses.mean_squared_error(labels, x, weight_matrix)

    # output end point error as evaluation metric
    firstlayer = labels[:, :, :, 0]
    bool_mask = tf.layers.flatten(tf.where(tf.abs(firstlayer) < 0.01, tf.zeros_like(firstlayer, dtype=tf.bool), tf.ones_like(firstlayer, dtype=tf.bool)))
    epe = tf.reduce_mean(tf.boolean_mask(tf.layers.flatten(tf.norm(labels - x, ord='euclidean', axis=3)), bool_mask))
    tf.summary.scalar('epe', epe)

    overall_epe = tf.reduce_mean(tf.norm(labels - x, ord='euclidean', axis=3))
    tf.summary.scalar('overall_epe', overall_epe)

    # image output
    tf.summary.image('output', tf.expand_dims(directionalFieldTF(x[0,:,:,0],x[0,:,:,1]),0))
    tf.summary.image('groundtruth', tf.expand_dims(directionalFieldTF(labels[0,:,:,0],labels[0,:,:,1]),0))

    # metrics for evaluation
    eval_metrics = {
        'epe':tf.metrics.mean_absolute_error(0.0, epe)
    }

    # solver
    global_step = tf.train.get_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=params['LEARNING_RATE'])
            train_op = optimizer.minimize(loss=loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics)

