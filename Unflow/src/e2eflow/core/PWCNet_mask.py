import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from .ops import CostVolumeLayer, WarpingLayer
import numpy as np


def pwcnet(im1, im2, option=[6, 5, 6, 'dense'], backward_flow=False):
    flows_fw = []
    flows_bw = []

    feature_list = [8, 16, 32, 48]
    [num_conv, num_concat, num_dilate, opt] = option

    def scoped_block():
        with tf.variable_scope('pwcnet_features'):
            conv1 = pwc_encoder(im1, num_conv, feature_list)
            conv2 = pwc_encoder(im2, num_conv, feature_list, reuse=True)

        with tf.variable_scope('pwcnet') as scope:
            flow_fw = pwc_decoder(conv1, conv2, num_conv, 4, num_concat, num_dilate, opt)
            flows_fw.append(flow_fw)

            if backward_flow:
                scope.reuse_variables()
                flow_bw = pwc_decoder(conv2, conv1, num_conv, 4, num_concat, num_dilate, opt)
                flows_bw.append(flow_bw)

    scoped_block()

    if backward_flow:
        return flows_fw, flows_bw
    return flows_fw


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)


def encoder_block(x, num, feature, reuse=None):
    x = slim.conv2d(x, feature, kernel_size=3, stride=2, scope='conv'+str(num), reuse=reuse)
    x = slim.conv2d(x, 16, kernel_size=3, stride=1, scope='conv'+str(num)+'_1', reuse=reuse)
    x = slim.conv2d(x, 16, kernel_size=3, stride=1, scope='conv'+str(num)+'_2', reuse=reuse)
    return x


def pwc_encoder(x, num, feature, reuse=None):
    with slim.arg_scope([slim.conv2d],
                        data_format='NHWC',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):

        convs = [x]
        for num_layer in range(num):
            x = encoder_block(x, num_layer+1, feature[num_layer], reuse=reuse)
            convs = [x] + convs
    return convs


def decoder_block(inputs, num, num_con, scale, md, num_dilate, opt):
    [c1, c2, up_flow, up_feat] = inputs
    feature_list = [64, 48, 32, 16]
    if up_flow is None and up_feat is None:
        x = CostVolumeLayer(c1, c2, md)
    else:
        warp = WarpingLayer(c2, up_flow)
        corr = CostVolumeLayer(c1, warp, md)
        x = tf.concat([corr, c1, up_flow, up_feat], axis=3)

    if num >= 2:
        for num_c in range(num_con):
            conv = slim.conv2d(x, feature_list[num_c], kernel_size=3, scope='corr'+str(num)+'_'+str(num_c), stride=1)
            x = tf.concat([conv, x], axis=3) if opt == 'dense' else conv

    flow = slim.conv2d(x, 2, kernel_size=3, scope='flow'+str(num), stride=1, activation_fn=None)

    if num == 0:
        return [flow, None, None]
    else:
        up_flow = slim.conv2d_transpose(flow, 2, kernel_size=4, scope='up_flow'+str(num), stride=2, activation_fn=None)
        up_feat = slim.conv2d_transpose(x, 2, kernel_size=4, scope='up_feat'+str(num), stride=2)
        return [flow, up_flow*scale, up_feat]


def pwc_decoder(conv1, conv2, num_layer, md, num_concat, num_dilate, opt):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NHWC',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        scale = [2.5, 5.0, 10.0, 20.0, None]
        flows = []
        up_flow = None
        up_feat = None
        for num in range(num_layer+1):
            inputs = [conv1[num], conv2[num], up_flow, up_feat]
            [flow, up_flow, up_feat] = decoder_block(inputs, num_layer-num, num_concat, scale[num], md, num_dilate, opt=opt)
            flows = [flow] + flows
        return flows
