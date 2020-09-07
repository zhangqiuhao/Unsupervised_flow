import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from .losses import multi_channels_to_grayscale, charbonnier_loss, length_sq
from .ops import get_grid
from .mask_network import masknet
from .ops import WarpingLayer
from .image_warp import image_warp


def _track_image(op, category, name):
    name = category + '/' + name
    tf.add_to_collection('train_images', tf.identity(op, name=name))


def compute_motion_loss(im_fw, flow_fw, im_bw, flow_bw, num_layer, mask_fw, mask_bw, border_mask=None):
    mask_fw_boolean = tf.cast(tf.nn.relu(tf.sign(mask_fw)), dtype=tf.bool)
    mask_bw_boolean = tf.cast(tf.nn.relu(tf.sign(mask_bw)), dtype=tf.bool)

    # Filter out points without detections
    bool_mask_fw = none_detection_mask(multi_channels_to_grayscale(im_fw, num_layer), border_mask)
    bool_mask_bw = none_detection_mask(multi_channels_to_grayscale(im_bw, num_layer), border_mask)

    bool_mask_fw = tf.logical_and(bool_mask_fw, mask_fw_boolean)
    bool_mask_bw = tf.logical_and(bool_mask_bw, mask_bw_boolean)

    bool_mask_fw_float32 = tf.cast(bool_mask_fw, dtype=tf.float32)
    bool_mask_bw_float32 = tf.cast(bool_mask_bw, dtype=tf.float32)

    # Estimate motion
    motion = motion_estimator(flow_fw, bool_mask_fw, border_mask)

    # Forward motion absolute l2 error
    motion_fw_wod = tf.multiply(bool_mask_fw_float32, motion)
    flow_fw_wod = tf.multiply(bool_mask_fw_float32, flow_fw)

    # Backward motion absolute l2 error
    motion_bw_wod = tf.multiply(bool_mask_bw_float32, -1.0 * motion)
    flow_bw_wod = tf.multiply(bool_mask_bw_float32, flow_bw)

    motion_losses = motion_loss(motion_fw_wod, flow_fw_wod, border_mask) + \
                    motion_loss(motion_bw_wod, flow_bw_wod, border_mask)

    mask_fw = tf.nn.sigmoid(mask_fw)
    mask_bw = tf.nn.sigmoid(mask_bw)
    motion_mask_loss = charbonnier_loss(1-mask_fw) + charbonnier_loss(1-mask_bw)

    mask_fw_boolean = tf.cast(mask_fw_boolean, dtype=tf.float32)
    mask_bw_boolean = tf.cast(mask_bw_boolean, dtype=tf.float32)

    _track_image(mask_fw_boolean, 'motion', 'mask_fw')
    _track_image(mask_bw_boolean, 'motion', 'mask_bw')

    return motion_losses, motion_mask_loss, motion


def none_detection_mask(im, mask=None):
    with tf.variable_scope('create_border_mask'):
        #extract pixels with value
        if mask is not None:
            im = tf.multiply(mask, im)
        im = tf.cast(im * 255.0, dtype=tf.uint8)
        zeros = tf.cast(tf.zeros_like(im), dtype=tf.bool)
        ones = tf.cast(tf.ones_like(im), dtype=tf.bool)
        return tf.stop_gradient(tf.where(im > 0, ones, zeros))


def motion_loss(motion, flow, mask=None):
    with tf.variable_scope('motion_loss'):
        if mask is not None:
            motion = tf.multiply(mask, motion)
            flow = tf.multiply(mask, flow)
        return charbonnier_loss(motion - flow)


def motion_estimator(flow, none_det_mask, mask):  #y height, x width
    with tf.variable_scope('motion_estimator'):
        _, h, w, _ = tf.unstack(tf.shape(flow))
        h = tf.cast(h, dtype=tf.float32)
        w = tf.cast(w, dtype=tf.float32)
        grid_b, grid_y, grid_x = get_grid(flow)

        grid_x = tf.cast(grid_x, tf.float32)
        grid_y = tf.cast(grid_y, tf.float32)

        #shift middle points
        #ones_tmp = tf.ones_like(grid_y)
        #shift_x = ones_tmp * w / 2.0
        #shift_y = ones_tmp * (h - 1.0)
        #grid_x = grid_x - shift_x
        #grid_y = shift_y - grid_y

        if mask is not None:
            flow = tf.multiply(mask, flow)

        #warp flow onto u and v
        warped_gx = tf.add(grid_x, flow[:, :, :, 0])
        warped_gy = tf.add(grid_y, flow[:, :, :, 1])

        elems = (grid_x, grid_y, warped_gx, warped_gy, none_det_mask)
        estimated_flow = tf.map_fn(calculate_flow, elems, dtype=tf.float32)

        return tf.stop_gradient(estimated_flow)


def calculate_flow(elements, sample=5):
    gx, gy, gx_flow, gy_flow, none_det_mask = elements
    none_det_mask = tf.reshape(none_det_mask, [-1])

    gx_masked = tf.expand_dims(tf.boolean_mask(tf.reshape(gx, [-1]), none_det_mask), 1)
    gy_masked = tf.expand_dims(tf.boolean_mask(tf.reshape(gy, [-1]), none_det_mask), 1)

    gx_flow_masked = tf.expand_dims(tf.boolean_mask(tf.reshape(gx_flow, [-1]), none_det_mask), 1)
    gy_flow_masked = tf.expand_dims(tf.boolean_mask(tf.reshape(gy_flow, [-1]), none_det_mask), 1)

    matrix_im1 = tf.concat([gx_masked, gy_masked], axis=1)
    matrix_im2 = tf.concat([gx_flow_masked, gy_flow_masked], axis=1)

    mat_stack = tf.random_shuffle(tf.stack([matrix_im1, matrix_im2], axis=1))
    matrix_im1 = tf.reshape(tf.transpose(mat_stack[0:sample, 0, :]), [2, sample])
    matrix_im2 = tf.reshape(tf.transpose(mat_stack[0:sample, 1, :]), [2, sample])

    R, c, t = ralign(matrix_im1, matrix_im2)

    origin_pos = tf.concat([tf.expand_dims(gx, 0), tf.expand_dims(gy, 0)], 0)
    flatted_pos = tf.reshape(origin_pos, [2, -1])
    rotated_pos = tf.transpose(tf.reshape(tf.matmul(R, flatted_pos), tf.shape(origin_pos)), [1, 2, 0])

    moved_pos = rotated_pos + tf.tile(tf.transpose(tf.expand_dims(t, -1)), tf.shape(tf.expand_dims(gx, -1)))
    moved_pos = moved_pos - tf.transpose(origin_pos, [1, 2, 0])

    return moved_pos


def ralign(X, Y):
    m, n = tf.unstack(tf.shape(X))

    mx = tf.expand_dims(tf.reduce_mean(X, 1), 1)
    my = tf.expand_dims(tf.reduce_mean(Y, 1), 1)

    Xc = X - tf.tile(mx, [1, n])
    Yc = Y - tf.tile(my, [1, n])

    sx = tf.reduce_mean(tf.reduce_sum(tf.multiply(Xc, Xc), 0))
    Sxy = tf.matmul(Yc, tf.transpose(Xc)) / tf.cast(n, tf.float32)

    D, U, V = tf.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    S = tf.eye(m, dtype=tf.float32)

    R = tf.matmul(tf.matmul(U, S), V)
    c = tf.trace(tf.matmul(tf.diag(D), S)) / sx
    t = my - c * tf.matmul(R, mx)

    return R, c, t
