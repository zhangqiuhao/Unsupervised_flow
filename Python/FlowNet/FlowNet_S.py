import tensorflow as tf
from tools.PlotDirectionalField import directionalFieldTF
from tensorflow.python.ops.init_ops import Initializer
import numpy as np


class LoadInitializer(Initializer):

    def __init__(self, dtype=tf.float32, name=''):
        self.dtype = tf.as_dtype(dtype)
        self.filename = '/home/klein/U/Masterarbeit/FlowNet/extracted_weights/flownet_s/' + name + '.npy'

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        data = np.load(self.filename)
        tensor = tf.reshape(tf.convert_to_tensor(data, dtype=dtype), shape)

        return tensor

    def get_config(self):
        return {"dtype": self.dtype.name}


def model_fn(features, labels, mode, params):

    # resizing:
    features = tf.image.resize_images(features, [64, 1024])
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.image.resize_images(labels, [64, 1024])

    x = features

    if params['FIRSTRUN']:

        x = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='same', name='conv1', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='conv1_weights'),  bias_initializer=LoadInitializer(name='conv1_biases'))
        x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='same', name='conv2', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='conv2_weights'), bias_initializer=LoadInitializer(name='conv2_biases'))
        x_conv2 = x
        x = tf.layers.conv2d(x, filters=256, kernel_size=5, strides=2, padding='same', name='conv3', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='conv3_weights'), bias_initializer=LoadInitializer(name='conv3_biases'))
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding='same', name='conv3_1', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='conv3_1_weights'), bias_initializer=LoadInitializer(name='conv3_1_biases'))
        x_conv3_1 = x
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', name='conv4', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='conv4_weights'), bias_initializer=LoadInitializer(name='conv4_biases'))
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', name='conv4_1', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='conv4_1_weights'), bias_initializer=LoadInitializer(name='conv4_1_biases'))
        x_conv4_1 = x
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', name='conv5', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='conv5_weights'), bias_initializer=LoadInitializer(name='conv5_biases'))
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', name='conv5_1', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='conv5_1_weights'), bias_initializer=LoadInitializer(name='conv5_1_biases'))
        x_conv5_1 = x
        x = tf.layers.conv2d(x, filters=1024, kernel_size=3, strides=2, padding='same', name='conv6', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='conv6_weights'), bias_initializer=LoadInitializer(name='conv6_biases'))
        x = tf.layers.conv2d(x, filters=1024, kernel_size=3, strides=1, padding='same', name='conv6_1', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='conv6_1_weights'), bias_initializer=LoadInitializer(name='conv6_1_biases'))

        # refinement
        predict_6 = tf.layers.conv2d(x, filters=2, kernel_size=3, padding='same', kernel_initializer=LoadInitializer(name='predict_flow6_weights'), bias_initializer=LoadInitializer(name='predict_flow6_biases'))
        x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=4, strides=2, padding='same', name='deconv5', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='deconv5_weights'), bias_initializer=LoadInitializer(name='deconv5_biases'))
        upsample_6to5 = tf.layers.conv2d_transpose(predict_6, filters=2, kernel_size=4, strides=2, padding='same', kernel_initializer=LoadInitializer(name='upsample_flow6to5_weights'), bias_initializer=LoadInitializer(name='upsample_flow6to5_biases'))
        x = tf.concat([x_conv5_1, x, upsample_6to5], axis=3, name='concat_1')

        predict_5 = tf.layers.conv2d(x, filters=2, kernel_size=3, padding='same', kernel_initializer=LoadInitializer(name='predict_flow5_weights'), bias_initializer=LoadInitializer(name='predict_flow5_biases'))
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='same', name='deconv4', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='deconv4_weights'), bias_initializer=LoadInitializer(name='deconv4_biases'))
        upsample_5to4 = tf.layers.conv2d_transpose(predict_5, filters=2, kernel_size=4, strides=2, padding='same', kernel_initializer=LoadInitializer(name='upsample_flow5to4_weights'), bias_initializer=LoadInitializer(name='upsample_flow5to4_biases'))
        x = tf.concat([x_conv4_1, x, upsample_5to4], axis=3, name='concat_2')

        predict_4 = tf.layers.conv2d(x, filters=2, kernel_size=3, padding='same', kernel_initializer=LoadInitializer(name='predict_flow4_weights'), bias_initializer=LoadInitializer(name='predict_flow4_biases'))
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='same', name='deconv3', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='deconv3_weights'), bias_initializer=LoadInitializer(name='deconv3_biases'))
        upsample_4to3 = tf.layers.conv2d_transpose(predict_4, filters=2, kernel_size=4, strides=2, padding='same', kernel_initializer=LoadInitializer(name='upsample_flow4to3_weights'), bias_initializer=LoadInitializer(name='upsample_flow4to3_biases'))
        x = tf.concat([x_conv3_1, x, upsample_4to3], axis=3, name='concat_3')

        predict_3 = tf.layers.conv2d(x, filters=2, kernel_size=3, padding='same', kernel_initializer=LoadInitializer(name='predict_flow3_weights'), bias_initializer=LoadInitializer(name='predict_flow3_biases'))
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', name='deconv2', activation=tf.nn.relu, kernel_initializer=LoadInitializer(name='deconv2_weights'), bias_initializer=LoadInitializer(name='deconv2_biases'))
        upsample_3to2 = tf.layers.conv2d_transpose(predict_3, filters=2, kernel_size=4, strides=2, padding='same', kernel_initializer=LoadInitializer(name='upsample_flow3to2_weights'), bias_initializer=LoadInitializer(name='upsample_flow3to2_biases'))
        x = tf.concat([x_conv2, x, upsample_3to2], axis=3, name='concat_4')

    else:

        x = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='same', activation=tf.nn.relu, name='conv1')
        x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu, name='conv2')
        x_conv2 = x
        x = tf.layers.conv2d(x, filters=256, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu, name='conv3')
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='conv3_1')
        x_conv3_1 = x
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='conv4')
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='conv4_1')
        x_conv4_1 = x
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='conv5')
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='conv5_1')
        x_conv5_1 = x
        x = tf.layers.conv2d(x, filters=1024, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='conv6')
        x = tf.layers.conv2d(x, filters=1024, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='conv6_1')

        # refinement
        predict_6 = tf.layers.conv2d(x, filters=2, kernel_size=3,  padding='same')
        x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, name='deconv5')
        upsample_6to5 = tf.layers.conv2d_transpose(predict_6, filters=2, kernel_size=4, strides=2,  padding='same')
        x = tf.concat([x_conv5_1, x, upsample_6to5], axis=3, name='concat_1')

        predict_5 = tf.layers.conv2d(x, filters=2, kernel_size=3,  padding='same')
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, name='deconv4')
        upsample_5to4 = tf.layers.conv2d_transpose(predict_5, filters=2, kernel_size=4, strides=2,  padding='same')
        x = tf.concat([x_conv4_1, x, upsample_5to4], axis=3, name='concat_2')

        predict_4 = tf.layers.conv2d(x, filters=2, kernel_size=3,  padding='same')
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, name='deconv3')
        upsample_4to3 = tf.layers.conv2d_transpose(predict_4, filters=2, kernel_size=4, strides=2,  padding='same')
        x = tf.concat([x_conv3_1, x, upsample_4to3], axis=3, name='concat_3')

        predict_3 = tf.layers.conv2d(x, filters=2, kernel_size=3,  padding='same')
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, name='deconv2')
        upsample_3to2 = tf.layers.conv2d_transpose(predict_3, filters=2, kernel_size=4, strides=2,  padding='same')
        x = tf.concat([x_conv2, x, upsample_3to2], axis=3, name='concat_4')

    predict_2 = tf.layers.conv2d(x, filters=3, kernel_size=3, padding='same')

    x = predict_2
    x = tf.image.resize_bilinear(x, [64, 1024], align_corners=True)

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
