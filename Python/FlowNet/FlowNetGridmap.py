import tensorflow as tf
from tools.PlotDirectionalField import directionalFieldTF
from tensorflow.python.ops.init_ops import Initializer
import numpy as np


class LoadInitializer(Initializer):

    def __init__(self, dtype=tf.float32, name=''):
        self.dtype = tf.as_dtype(dtype)
        self.filename = '/home/klein/U/extracted_weights/flownet_s/' + name + '.npy'

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        data = np.load(self.filename)
        tensor = tf.reshape(tf.convert_to_tensor(data, dtype=dtype), shape)

        return tensor

    def get_config(self):
        return {"dtype": self.dtype.name}


def model_fn(features, labels, mode, params):
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # if not params['CROP_CENTER'] == params['IMG_SIZE']:
    #     features = tf.slice(features, begin=[0, int((params['IMG_SIZE']-params['CROP_CENTER'])/2), int((params['IMG_SIZE']-params['CROP_CENTER'])/2), 0], size=[-1, params['CROP_CENTER'], params['CROP_CENTER'], -1])
    #     if mode != tf.estimator.ModeKeys.PREDICT:
    #         labels = tf.slice(labels, begin=[0, int((params['IMG_SIZE'] - params['CROP_CENTER']) / 2), int((params['IMG_SIZE'] - params['CROP_CENTER']) / 2), 0], size=[-1, params['CROP_CENTER'], params['CROP_CENTER'], -1])

    # resizing:
    features = tf.image.resize_images(features, [params['RESIZE'], params['RESIZE']])
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.image.resize_images(labels, [params['RESIZE'], params['RESIZE']])

    x = tf.stack([features[:,:,:,0],features[:,:,:,0],features[:,:,:,0],features[:,:,:,1],features[:,:,:,1],features[:,:,:,1]],axis=3)

    x = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='same', name='conv1', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='conv1_weights'),  bias_initializer=LoadInitializer(name='conv1_biases'))
    x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='same', name='conv2', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='conv2_weights'), bias_initializer=LoadInitializer(name='conv2_biases'))
    x_conv2 = x
    x = tf.layers.conv2d(x, filters=256, kernel_size=5, strides=2, padding='same', name='conv3', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='conv3_weights'), bias_initializer=LoadInitializer(name='conv3_biases'))
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding='same', name='conv3_1', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='conv3_1_weights'), bias_initializer=LoadInitializer(name='conv3_1_biases'))
    x_conv3_1 = x
    x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', name='conv4', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='conv4_weights'), bias_initializer=LoadInitializer(name='conv4_biases'))
    x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', name='conv4_1', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='conv4_1_weights'), bias_initializer=LoadInitializer(name='conv4_1_biases'))
    x_conv4_1 = x
    x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', name='conv5', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='conv5_weights'), bias_initializer=LoadInitializer(name='conv5_biases'))
    x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', name='conv5_1', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='conv5_1_weights'), bias_initializer=LoadInitializer(name='conv5_1_biases'))
    x_conv5_1 = x
    x = tf.layers.conv2d(x, filters=1024, kernel_size=3, strides=2, padding='same', name='conv6', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='conv6_weights'), bias_initializer=LoadInitializer(name='conv6_biases'))
    x = tf.layers.conv2d(x, filters=1024, kernel_size=3, strides=1, padding='same', name='conv6_1', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='conv6_1_weights'), bias_initializer=LoadInitializer(name='conv6_1_biases'))

    predict_6 = tf.layers.conv2d(x, filters=2, kernel_size=3, padding='same', kernel_initializer=LoadInitializer(name='predict_flow6_weights'), bias_initializer=LoadInitializer(name='predict_flow6_biases'))
    x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=4, strides=2, padding='same', name='deconv5', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='deconv5_weights'), bias_initializer=LoadInitializer(name='deconv5_biases'))
    upsample_6to5 = tf.layers.conv2d_transpose(predict_6, filters=2, kernel_size=4, strides=2, padding='same', kernel_initializer=LoadInitializer(name='upsample_flow6to5_weights'), bias_initializer=LoadInitializer(name='upsample_flow6to5_biases'))
    x = tf.concat([x_conv5_1, x, upsample_6to5], axis=3, name='concat_1')

    predict_5 = tf.layers.conv2d(x, filters=2, kernel_size=3, padding='same', kernel_initializer=LoadInitializer(name='predict_flow5_weights'), bias_initializer=LoadInitializer(name='predict_flow5_biases'))
    x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='same', name='deconv4', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='deconv4_weights'), bias_initializer=LoadInitializer(name='deconv4_biases'))
    upsample_5to4 = tf.layers.conv2d_transpose(predict_5, filters=2, kernel_size=4, strides=2, padding='same', kernel_initializer=LoadInitializer(name='upsample_flow5to4_weights'), bias_initializer=LoadInitializer(name='upsample_flow5to4_biases'))
    x = tf.concat([x_conv4_1, x, upsample_5to4], axis=3, name='concat_2')

    predict_4 = tf.layers.conv2d(x, filters=2, kernel_size=3, padding='same', kernel_initializer=LoadInitializer(name='predict_flow4_weights'), bias_initializer=LoadInitializer(name='predict_flow4_biases'))
    x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='same', name='deconv3', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='deconv3_weights'), bias_initializer=LoadInitializer(name='deconv3_biases'))
    upsample_4to3 = tf.layers.conv2d_transpose(predict_4, filters=2, kernel_size=4, strides=2, padding='same', kernel_initializer=LoadInitializer(name='upsample_flow4to3_weights'), bias_initializer=LoadInitializer(name='upsample_flow4to3_biases'))
    x = tf.concat([x_conv3_1, x, upsample_4to3], axis=3, name='concat_3')

    predict_3 = tf.layers.conv2d(x, filters=2, kernel_size=3, padding='same', kernel_initializer=LoadInitializer(name='predict_flow3_weights'), bias_initializer=LoadInitializer(name='predict_flow3_biases'))
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', name='deconv2', activation=tf.nn.leaky_relu, kernel_initializer=LoadInitializer(name='deconv2_weights'), bias_initializer=LoadInitializer(name='deconv2_biases'))
    upsample_3to2 = tf.layers.conv2d_transpose(predict_3, filters=2, kernel_size=4, strides=2, padding='same', kernel_initializer=LoadInitializer(name='upsample_flow3to2_weights'), bias_initializer=LoadInitializer(name='upsample_flow3to2_biases'))
    x = tf.concat([x_conv2, x, upsample_3to2], axis=3, name='concat_4')

    predict_2 = tf.layers.conv2d(x, filters=2, kernel_size=3, padding='same', kernel_initializer=LoadInitializer(name='predict_flow2_weights'), bias_initializer=LoadInitializer(name='predict_flow2_biases'))

    x = predict_2
    x = tf.image.resize_bilinear(x, [params['RESIZE'], params['RESIZE']], align_corners=True)

    # for prediction only:
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"predictions": x})


    # calculate loss:
    mask = tf.stack([features[:, :, :, 0], features[:, :, :, 0]], axis=3)
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
