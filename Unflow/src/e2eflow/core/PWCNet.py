import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import numpy as np

from ..ops import correlation


def pwcnet(im1, im2, backward_flow=False):
    num_batch, height, width, _ = tf.unstack(tf.shape(im1))

    with tf.variable_scope('pwc_encoder'):
        flows_fw = []
        flows_bw = []
        conv1 = pwc_encoder(im1)
        conv2 = pwc_encoder(im2, reuse=True)

    with tf.variable_scope('pwc_decoder') as scope:
        flow_fw = pwc_decoder(conv1, conv2, 4)
        flows_fw.append(flow_fw)

        if backward_flow:
            scope.reuse_variables()
            flow_bw = pwc_decoder(conv2, conv1, 4)
            flows_bw.append(flow_bw)

    if backward_flow:
        return flows_fw, flows_bw
    return flows_fw


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)


def nhwc_to_nchw(tensors):
    return [tf.transpose(t, [0, 3, 1, 2]) for t in tensors]


def nchw_to_nhwc(tensors):
    return [tf.transpose(t, [0, 2, 3, 1]) for t in tensors]


def pwc_encoder(im, reuse=None):
    im = nhwc_to_nchw([im])[0]
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
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


def pwc_decoder(conv1, conv2 , md):
    [c16, c15, c14, c13, c12] = conv1  #NCHW
    [c26, c25, c24, c23, c22] = conv2  #NCHW

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):

        corr6 = correlation(c16, c26,
                            pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1)
        corr6 = tf.nn.leaky_relu(corr6, alpha=0.1)

        x = tf.concat([slim.conv2d(corr6, 128, kernel_size=3, scope='corr6_1', stride=1), corr6], 1)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr6_2', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 96, kernel_size=3, scope='corr6_3', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 64, kernel_size=3, scope='corr6_4', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 32, kernel_size=3, scope='corr6_5', stride=1), x], 1)
        flow6 = slim.conv2d(x, 2, kernel_size=3, scope='flow6', stride=1, activation_fn=None)
        up_flow6 = slim.conv2d_transpose(flow6, 2, kernel_size=4, scope='up_flow6', stride=2, activation_fn=None)
        up_feat6 = slim.conv2d_transpose(x, 2, kernel_size=4, scope='up_feat6', stride=2, activation_fn=None)

        warp5 = image_warp(c25, up_feat6*0.625)
        corr5 = correlation(c15, warp5,
                            pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1)
        corr5 = tf.nn.leaky_relu(corr5, alpha=0.1)
        x = tf.concat([corr5, c15, up_flow6, up_feat6], 1)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr5_1', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr5_2', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 96, kernel_size=3, scope='corr5_3', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 64, kernel_size=3, scope='corr5_4', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 32, kernel_size=3, scope='corr5_5', stride=1), x], 1)
        flow5 = slim.conv2d(x, 2, kernel_size=3, scope='flow5', stride=1, activation_fn=None)
        up_flow5 = slim.conv2d_transpose(flow5, 2, kernel_size=4, scope='up_flow5', stride=2, activation_fn=None)
        up_feat5 = slim.conv2d_transpose(x, 2, kernel_size=4, scope='up_feat5', stride=2, activation_fn=None)

        warp4 = image_warp(c24, up_feat5*1.25)
        corr4 = correlation(c14, warp4,
                            pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1)
        corr4 = tf.nn.leaky_relu(corr4, alpha=0.1)
        x = tf.concat([corr4, c14, up_flow5, up_feat5], 1)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr4_1', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr4_2', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 96, kernel_size=3, scope='corr4_3', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 64, kernel_size=3, scope='corr4_4', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 32, kernel_size=3, scope='corr4_5', stride=1), x], 1)
        flow4 = slim.conv2d(x, 2, kernel_size=3, scope='flow4', stride=1, activation_fn=None)
        up_flow4 = slim.conv2d_transpose(flow4, 2, kernel_size=4, scope='up_flow4', stride=2, activation_fn=None)
        up_feat4 = slim.conv2d_transpose(x, 2, kernel_size=4, scope='up_feat4', stride=2, activation_fn=None)

        warp3 = image_warp(c23, up_feat4*2.5)
        corr3 = correlation(c13, warp3,
                            pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1)
        corr3 = tf.nn.leaky_relu(corr3, alpha=0.1)
        x = tf.concat([corr3, c13, up_flow4, up_feat4], 1)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr3_1', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr3_2', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 96, kernel_size=3, scope='corr3_3', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 64, kernel_size=3, scope='corr3_4', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 32, kernel_size=3, scope='corr3_5', stride=1), x], 1)
        flow3 = slim.conv2d(x, 2, kernel_size=3, scope='flow3', stride=1, activation_fn=None)
        up_flow3 = slim.conv2d_transpose(flow3, 2, kernel_size=4, scope='up_flow3', stride=2, activation_fn=None)
        up_feat3 = slim.conv2d_transpose(x, 2, kernel_size=4, scope='up_feat3', stride=2, activation_fn=None)

        warp2 = image_warp(c22, up_feat3*5.0)
        corr2 = correlation(c12, warp2,
                            pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1)
        corr2 = tf.nn.leaky_relu(corr2, alpha=0.1)
        x = tf.concat([corr2, c12, up_flow3, up_feat3], 1)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr2_1', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 128, kernel_size=3, scope='corr2_2', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 96, kernel_size=3, scope='corr2_3', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 64, kernel_size=3, scope='corr2_4', stride=1), x], 1)
        x = tf.concat([slim.conv2d(x, 32, kernel_size=3, scope='corr2_5', stride=1), x], 1)
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

    return nchw_to_nhwc(flows)


def image_warp(im, flow):
    """Performs a backward warp of an image using the predicted flow.

    Args:
        im: Batch of images. [num_batch, channels, height, width]
        flow: Batch of flow vectors. [num_batch, 2, height, width]
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    im = tf.transpose(im, [0, 2, 3, 1])
    flow = tf.transpose(flow, [0, 2, 3, 1])

    with tf.variable_scope('image_warp'):

        num_batch, height, width, channels = tf.unstack(tf.shape(im))
        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # We have to flatten our tensors to vectorize the interpolation
        im_flat = tf.reshape(im, [-1, channels])
        flow_flat = tf.reshape(flow, [-1, 2])

        # Floor the flow, as the final indices are integers
        # The fractional part is used to control the bilinear interpolation.
        flow_floor = tf.to_int32(tf.floor(flow_flat))
        bilinear_weights = flow_flat - tf.floor(flow_flat)

        # Construct base indices which are displaced with the flow
        pos_x = tf.tile(tf.range(width), [height * num_batch])
        grid_y = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
        pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])

        x = flow_floor[:, 0]
        y = flow_floor[:, 1]
        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = tf.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
        wb = tf.expand_dims((1 - xw) * yw, 1) # bottom left pixel
        wc = tf.expand_dims(xw * (1 - yw), 1) # top right pixel
        wd = tf.expand_dims(xw * yw, 1) # bottom right pixel

        x0 = pos_x + x
        x1 = x0 + 1
        y0 = pos_y + y
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim1 = width * height
        batch_offsets = tf.range(num_batch) * dim1
        base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
        base = tf.reshape(base_grid, [-1])

        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        warped = tf.reshape(warped_flat, [num_batch, height, width, channels])

        return tf.transpose(warped, [0, 3, 1, 2])
