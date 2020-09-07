import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from .ops import CostVolumeLayer, WarpingLayer
import numpy as np


def pwcnet(im1, im2, option='', backward_flow=False):
    num_batch, height, width, channels = tf.unstack(tf.shape(im1))
    flows_fw = []
    flows_bw = []

    def scoped_block():
        with tf.variable_scope('pwc_features'):
            conv1 = pwc_encoder(im1)
            conv2 = pwc_encoder(im2, reuse=True)

        with tf.variable_scope('pwc') as scope:
            flow_fw = pwc_decoder(conv1, conv2, 4)
            flows_fw.append(flow_fw)

            if backward_flow:
                scope.reuse_variables()
                flow_bw = pwc_decoder(conv2, conv1, 4)
                flows_bw.append(flow_bw)

    scoped_block()

    if backward_flow:
        return flows_fw, flows_bw
    return flows_fw


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)


def pwc_encoder(im, reuse=None):
    with slim.arg_scope([slim.conv2d],
                        data_format='NHWC',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(im, 16, kernel_size=3, stride=2, scope='conv1', reuse=reuse)
        conv1_1 = slim.conv2d(conv1, 16, kernel_size=3, stride=1, scope='conv1_1', reuse=reuse)
        conv1_2 = slim.conv2d(conv1_1, 16, kernel_size=3, stride=1, scope='conv1_2', reuse=reuse)

        conv2 = slim.conv2d(conv1_2, 32, kernel_size=3, stride=2, scope='conv2', reuse=reuse)
        conv2_1 = slim.conv2d(conv2, 32, kernel_size=3, stride=1, scope='conv2_1', reuse=reuse)
        conv2_2 = slim.conv2d(conv2_1, 32, kernel_size=3, stride=1, scope='conv2_2', reuse=reuse)

        conv3 = slim.conv2d(conv2_2, 64, kernel_size=3, stride=2, scope='conv3', reuse=reuse)
        conv3_1 = slim.conv2d(conv3, 64, kernel_size=3, stride=1, scope='conv3_1', reuse=reuse)
        conv3_2 = slim.conv2d(conv3_1, 64, kernel_size=3, stride=1, scope='conv3_2', reuse=reuse)

        conv4 = slim.conv2d(conv3_2, 96, kernel_size=3, stride=2, scope='conv4', reuse=reuse)
        conv4_1 = slim.conv2d(conv4, 96, kernel_size=3, stride=1, scope='conv4_1', reuse=reuse)
        conv4_2 = slim.conv2d(conv4_1, 96, kernel_size=3, stride=1, scope='conv4_2', reuse=reuse)

        conv5 = slim.conv2d(conv4_2, 128, kernel_size=3, stride=2, scope='conv5', reuse=reuse)
        conv5_1 = slim.conv2d(conv5, 128, kernel_size=3, stride=1, scope='conv5_1', reuse=reuse)
        conv5_2 = slim.conv2d(conv5_1, 128, kernel_size=3, stride=1, scope='conv5_2', reuse=reuse)

        conv6 = slim.conv2d(conv5_2, 196, kernel_size=3, stride=2, scope='conv6', reuse=reuse)
        conv6_1 = slim.conv2d(conv6, 196, kernel_size=3, stride=1, scope='conv6_1', reuse=reuse)
        conv6_2 = slim.conv2d(conv6_1, 196, kernel_size=3, stride=1, scope='conv6_2', reuse=reuse)

        convs = [conv6_2, conv5_2, conv4_2, conv3_2, conv2_2]

    return convs


def pwc_decoder(conv1, conv2, md):
    [c16, c15, c14, c13, c12] = conv1  #NHWC
    [c26, c25, c24, c23, c22] = conv2  #NHWC

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NHWC',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):

        #layer 6
        x = CostVolumeLayer(c16, c26, md)

        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr6_1', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr6_2', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 96, kernel_size=3, scope='corr6_3', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 64, kernel_size=3, scope='corr6_4', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 32, kernel_size=3, scope='corr6_5', stride=1), x], 3)

        flow6 = slim.conv2d(x, 2, kernel_size=3, scope='flow6', stride=1, activation_fn=None)

        up_flow6 = slim.conv2d_transpose(flow6, 2, kernel_size=4, scope='up_flow6', stride=2, activation_fn=None)
        up_feat6 = slim.conv2d_transpose(x, 2, kernel_size=4, scope='up_feat6', stride=2)

        #layer 5
        warp5 = WarpingLayer(c25, up_flow6*0.625)
        corr5 = CostVolumeLayer(c15, warp5, md)

        x = tf.concat([corr5, c15, up_flow6, up_feat6], 3)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr5_1', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr5_2', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 96, kernel_size=3, scope='corr5_3', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 64, kernel_size=3, scope='corr5_4', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 32, kernel_size=3, scope='corr5_5', stride=1), x], 3)

        flow5 = slim.conv2d(x, 2, kernel_size=3, scope='flow5', stride=1, activation_fn=None)

        up_flow5 = slim.conv2d_transpose(flow5, 2, kernel_size=4, scope='up_flow5', stride=2, activation_fn=None)
        up_feat5 = slim.conv2d_transpose(x, 2, kernel_size=4, scope='up_feat5', stride=2)

        #layer 4
        warp4 = WarpingLayer(c24, up_flow5*1.25)
        corr4 = CostVolumeLayer(c14, warp4, md)

        x = tf.concat([corr4, c14, up_flow5, up_feat5], 3)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr4_1', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr4_2', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 96, kernel_size=3, scope='corr4_3', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 64, kernel_size=3, scope='corr4_4', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 32, kernel_size=3, scope='corr4_5', stride=1), x], 3)

        flow4 = slim.conv2d(x, 2, kernel_size=3, scope='flow4', stride=1, activation_fn=None)

        up_flow4 = slim.conv2d_transpose(flow4, 2, kernel_size=4, scope='up_flow4', stride=2, activation_fn=None)
        up_feat4 = slim.conv2d_transpose(x, 2, kernel_size=4, scope='up_feat4', stride=2)

        #layer 3
        warp3 = WarpingLayer(c23, up_flow4*2.5)
        corr3 = CostVolumeLayer(c13, warp3, md)

        x = tf.concat([corr3, c13, up_flow4, up_feat4], 3)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr3_1', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr3_2', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 96, kernel_size=3, scope='corr3_3', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 64, kernel_size=3, scope='corr3_4', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 32, kernel_size=3, scope='corr3_5', stride=1), x], 3)

        flow3 = slim.conv2d(x, 2, kernel_size=3, scope='flow3', stride=1, activation_fn=None)

        up_flow3 = slim.conv2d_transpose(flow3, 2, kernel_size=4, scope='up_flow3', stride=2, activation_fn=None)
        up_feat3 = slim.conv2d_transpose(x, 2, kernel_size=4, scope='up_feat3', stride=2)

        #layer 2
        warp2 = WarpingLayer(c22, up_flow3*5.0)
        corr2 = CostVolumeLayer(c12, warp2, md)

        x = tf.concat([corr2, c12, up_flow3, up_feat3], 3)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr2_1', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr2_2', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 96, kernel_size=3, scope='corr2_3', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 64, kernel_size=3, scope='corr2_4', stride=1), x], 3)
        x = tf.concat([slim.conv2d(x, 32, kernel_size=3, scope='corr2_5', stride=1), x], 3)

        flow2 = slim.conv2d(x, 2, kernel_size=3, scope='flow2', stride=1, activation_fn=None)

        x = slim.conv2d(x, 128, kernel_size=3, scope='diconv1', stride=1, rate=1)
        x = slim.conv2d(x, 128, kernel_size=3, scope='diconv2', stride=1, rate=2)
        x = slim.conv2d(x, 128, kernel_size=3, scope='diconv3', stride=1, rate=4)
        x = slim.conv2d(x, 96, kernel_size=3, scope='diconv4', stride=1, rate=8)
        x = slim.conv2d(x, 64, kernel_size=3, scope='diconv5', stride=1, rate=16)
        x = slim.conv2d(x, 32, kernel_size=3, scope='diconv6', stride=1, rate=1)
        x = slim.conv2d(x, 2, kernel_size=3, scope='diconv7', stride=1, activation_fn=None)

        flow2 = flow2 + x

        flows = [flow2, flow3, flow4, flow5, flow6]

        return flows
