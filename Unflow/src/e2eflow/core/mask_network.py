import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers


def masknet(motion_fw, flow_fw):
    with tf.variable_scope('mask_network'):
        motion_mask_fw = mask_creator(motion_fw, flow_fw)

    return motion_mask_fw


def mask_creator(motion, flow):
    with slim.arg_scope([slim.conv2d],
                        data_format='NHWC',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=None):

        def feature_extractor(x, reuse=None):
            x = slim.conv2d(x, 16, kernel_size=3, stride=1, scope='mot_est_feat_1', reuse=reuse)
            x = slim.conv2d(x, 32, kernel_size=3, stride=1, scope='mot_est_feat_2', reuse=reuse)
            x = slim.conv2d(x, 64, kernel_size=1, stride=1, scope='mot_est_feat_3', reuse=reuse)
            return x

        motion_feature = feature_extractor(motion)
        flow_feature = feature_extractor(flow, reuse=True)

        diff = tf.abs(motion_feature - flow_feature)
        diff = slim.conv2d(diff, 16, kernel_size=1, stride=1, scope='mask_conv_1')
        diff = slim.conv2d(diff, 32, kernel_size=1, stride=1, scope='mask_conv_2')
        diff = slim.conv2d(diff, 64, kernel_size=1, stride=1, scope='mask_conv_3')

        mask = slim.conv2d(diff, 1, kernel_size=1, stride=1, scope='mask_output')
        mask = tf.nn.relu(tf.sign(mask))
    return 1-mask
