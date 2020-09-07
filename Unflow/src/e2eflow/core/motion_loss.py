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


def compute_motion_loss(im_fw, flow_fw, im_bw, flow_bw, num_layer, border_mask=None):
    # Filter out points without detections
    bool_mask_fw = none_detection_mask(multi_channels_to_grayscale(im_fw, num_layer), border_mask)
    bool_mask_bw = none_detection_mask(multi_channels_to_grayscale(im_bw, num_layer), border_mask)
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

    mag_sq = length_sq(motion_fw_wod) + length_sq(motion_bw_wod)
    occ_thresh = 0.01 * mag_sq + 0.5

    diff_motion_fw = motion_fw_wod - flow_fw_wod
    diff_motion_bw = motion_bw_wod - flow_bw_wod

    #motion_mask_fw = tf.cast(length_sq(diff_motion_fw) < occ_thresh, tf.float32)
    #motion_mask_bw = tf.cast(length_sq(diff_motion_bw) < occ_thresh, tf.float32)

    motion_mask_fw = 1.0-tf.nn.tanh(length_sq(diff_motion_fw))
    motion_mask_bw = 1.0-tf.nn.tanh(length_sq(diff_motion_bw))

    #motion_mask_fw = masknet(motion_fw_wod, flow_fw_wod)
    #motion_mask_bw = WarpingLayer(motion_mask_fw, flow_bw, warp='nearest')

    motion_fw_wod = tf.multiply(motion_mask_fw, motion_fw_wod)
    flow_fw_wod = tf.multiply(motion_mask_fw, flow_fw_wod)
    motion_bw_wod = tf.multiply(motion_mask_bw, motion_bw_wod)
    flow_bw_wod = tf.multiply(motion_mask_bw, flow_bw_wod)

    motion_losses = motion_loss(motion_fw_wod, flow_fw_wod, border_mask) + \
                    motion_loss(motion_bw_wod, flow_bw_wod, border_mask)

    motion_mask_loss = charbonnier_loss(1-motion_mask_fw) + charbonnier_loss(1-motion_mask_bw)

    _track_image(motion_mask_fw, 'motion', 'mask_fw')
    _track_image(motion_mask_bw, 'motion', 'mask_bw')

    return motion_losses, motion_mask_loss, motion, motion_mask_fw


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

        if mask is not None:
            flow = tf.multiply(mask, flow)

        #warp flow onto u and v
        warped_gx = tf.add(grid_x, flow[:, :, :, 0])
        warped_gy = tf.add(grid_y, flow[:, :, :, 1])

        elems = (grid_x, grid_y, warped_gx, warped_gy, none_det_mask)
        estimated_flow = tf.map_fn(calculate_flow, elems, dtype=tf.float32)

        return tf.stop_gradient(estimated_flow)


def calculate_flow(elements):
    gx, gy, gx_flow, gy_flow, none_det_mask = elements
    none_det_mask = tf.reshape(none_det_mask, [-1])

    gx_masked = tf.expand_dims(tf.boolean_mask(tf.reshape(gx, [-1]), none_det_mask), 1)
    gy_masked = tf.expand_dims(tf.boolean_mask(tf.reshape(gy, [-1]), none_det_mask), 1)

    gx_flow_masked = tf.expand_dims(tf.boolean_mask(tf.reshape(gx_flow, [-1]), none_det_mask), 1)
    gy_flow_masked = tf.expand_dims(tf.boolean_mask(tf.reshape(gy_flow, [-1]), none_det_mask), 1)

    old_points = tf.transpose(tf.concat([gx_masked, gy_masked], axis=1))
    new_points = tf.transpose(tf.concat([gx_flow_masked, gy_flow_masked], axis=1))

    R, t = ralign(old_points, new_points)

    origin_pos = tf.concat([tf.expand_dims(gx, 0), tf.expand_dims(gy, 0)], 0)
    flatted_old_pos = tf.reshape(origin_pos, [2, -1])
    t = tf.matmul(t, tf.expand_dims(tf.ones_like(flatted_old_pos[0, :]), 0))
    moved_pos = tf.reshape(tf.add(tf.matmul(R, flatted_old_pos), t), tf.shape(origin_pos))
    moved_pos = tf.transpose(moved_pos - origin_pos, [1, 2, 0])

    return moved_pos


def ralign(old_points, new_points, number_iterations=10):
    n_ = tf.ones_like(old_points[0,:])
    vec_1_n = tf.expand_dims(n_, 0)
    weights = tf.matrix_diag(n_)

    for i in range(number_iterations):
        sum_p = tf.reduce_sum(weights)

        mx = tf.matmul(tf.matmul(old_points, weights), tf.expand_dims(n_, 1)) / sum_p
        my = tf.matmul(tf.matmul(new_points, weights), tf.expand_dims(n_, 1)) / sum_p

        Xc = old_points - tf.matmul(mx, vec_1_n)
        Yc = new_points - tf.matmul(my, vec_1_n)

        A = tf.matmul(tf.matmul(Yc, weights), tf.transpose(Xc)) / sum_p
        D, U, V = tf.linalg.svd(A, full_matrices=True, compute_uv=True)

        S = tf.matrix_diag(tf.ones_like(old_points[:,0]))

        R = tf.matmul(tf.matmul(U, S), tf.transpose(V))
        t = my - tf.matmul(R, mx)
        weights = squared_errors_to_weights(R, t, old_points, new_points, vec_1_n)
    return R, t


def squared_errors_to_weights(R, t, old_points, new_points, vec_1_n):
    t = tf.matmul(t, vec_1_n)
    moved_pose = tf.add(tf.matmul(R, old_points), t)
    err = tf.reduce_sum(tf.square(tf.subtract(moved_pose, new_points)), axis=0)
    max = tf.clip_by_value(tf.reduce_max(err), clip_value_min=1e-30, clip_value_max=1e100)
    weights = tf.matrix_diag(tf.divide(tf.subtract(max, err), max))
    return weights
