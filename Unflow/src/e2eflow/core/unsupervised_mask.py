import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.stats as st

from .augment import random_affine, random_photometric
from .flow_util import flow_to_color
from .util import resize_area, resize_bilinear
from .losses import compute_losses, create_border_mask
from .motion_loss_mask import compute_motion_loss
from ..ops import downsample
from .image_warp import image_warp
from .flownet import flownet, FLOW_SCALE     #edited
from .PWCNet_mask import pwcnet


def _track_loss(op, name):
    tf.add_to_collection('losses', tf.identity(op, name=name))


def _track_image(op, category, name):
    name = category + '/' + name
    tf.add_to_collection('train_images', tf.identity(op, name=name))


def unsupervised_loss(batch, start_iter, params, normalization=None, augment=True,
                      return_flow=False):
    params.get('')
    channel_mean = tf.constant(normalization[0]) / 255.0
    im1, im2 = batch

    im1 = im1 / 255.0
    im2 = im2 / 255.0

    im_shape = tf.shape(im1)[1:3]
    layers = params.get('layers').split(', ')
    num_layer = 0
    for layer in layers:
        if layer == 'rgb_cartesian':
            num_layer = num_layer + 3
        else:
            num_layer = num_layer + 1

    count_step = tf.Variable(start_iter, name='global_step', trainable=False, dtype=tf.int32)
    gauss_sigma = tf.ceil(5.0 - tf.cast(count_step, dtype=tf.float32) / 30000.0)
    mean = tf.constant(0.0, dtype=tf.float32)

    [im1, im2] = tf.cond(gauss_sigma > 0,
                         lambda: gaussian_blur(im1, im2, gauss_sigma*4+1, mean, gauss_sigma, num_layer),
                         lambda: gaussian_blur(im1, im2, 1.0*4+1, mean, 1.0, num_layer)
                         )

    # Data & mask augmentation
    border_mask = create_border_mask(im1, 0.1)

    if augment:
        im1_geo, im2_geo, border_mask_global = random_affine(
            [im1, im2, border_mask],
            horizontal_flipping=True
            )

        im2_geo, border_mask_local = random_affine(
            [im2_geo, border_mask], max_translation_x=0.05,
            max_translation_y=0.05, max_rotation=10.0
        )
        border_mask = border_mask_local * border_mask_global

        im1_photo, im2_photo = [tf.clip_by_value(im1_geo, 0.0, 1.0), tf.clip_by_value(im2_geo, 0.0, 1.0)]
    else:
        im1_geo, im2_geo = im1, im2
        im1_photo, im2_photo = im1, im2

    # Images for loss comparisons with values in [0, 1] (scale to original using * 255)
    im1_norm = im1_photo
    im2_norm = im2_photo
    # Images for neural network input with mean-zero values in [-1, 1]
    im1_photo = im1_photo - channel_mean
    im2_photo = im2_photo - channel_mean

    network = params.get('network')
    full_resolution = params.get('full_res')

    if network == 'pwcnet':
        num_conv = int(params.get('num_conv'))
        num_concat = int(params.get('num_concat'))
        num_dilate = int(params.get('num_dilate'))
        opt = params.get('opt')
        flows_fw, flows_bw, masks_fw, masks_bw = pwcnet(im1_photo, im2_photo,
                                                        option=[num_conv, num_concat, num_dilate, opt],
                                                        backward_flow=True)
    else:
        flownet_spec = params.get('flownet', 'S')
        train_all = params.get('train_all')

        flows_fw, flows_bw, masks_fw, masks_bw = flownet(im1_photo, im2_photo, num_layer,
                                                         flownet_spec=flownet_spec,
                                                         backward_flow=True,
                                                         train_all=train_all)

    flows_fw = flows_fw[-1]
    flows_bw = flows_bw[-1]
    masks_fw = masks_fw[-1]
    masks_bw = masks_bw[-1]

    # -------------------------------------------------------------------------
    # Losses
    # REGISTER ALL POSSIBLE LOSS TERMS
    LOSSES = ['sym', 'fb', 'grad', 'ternary', 'photo', 'smooth_1st', 'smooth_2nd']
    MASK_LOSSES = ['occ']

    im1_s = downsample(im1_norm, 4)
    im2_s = downsample(im2_norm, 4)
    mask_s = downsample(border_mask, 4)
    final_flow_scale = FLOW_SCALE
    final_flow_fw = tf.image.resize_bilinear(flows_fw[0], im_shape) * final_flow_scale * 4
    final_flow_bw = tf.image.resize_bilinear(flows_bw[0], im_shape) * final_flow_scale * 4

    combined_losses = dict()
    for loss in LOSSES:
        combined_losses[loss] = 0.0

    if params.get('motion_weight'):
        motion_loss, motion_mask_loss, motion_fw = \
            compute_motion_loss(im1_s, flows_fw[0]*final_flow_scale, im2_s, flows_bw[0]*final_flow_scale, num_layer,
                                masks_fw[0], masks_bw[0],
                                border_mask=mask_s if params.get('border_mask') else None)
        combined_losses['motion'] = 200.0 * motion_loss
        combined_losses['motion_mask'] = motion_mask_loss
        _track_image(flow_to_color(motion_fw), 'Motion', 'Estimated_fw')

    layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]
    layer_patch_distances = [3, 2, 2, 1, 1]

    _track_image(flow_to_color(final_flow_fw), 'flow', 'forward')
    _track_image(im1_s[..., 0:3], 'img_input', 'augmented1')
    _track_image(im2_s[..., 0:3], 'img_input', 'augmented2')

    if params.get('pyramid_loss'):
        flow_enum = enumerate(zip(flows_fw, flows_bw))
    else:
        flow_enum = [(0, (flows_fw[0], flows_bw[0]))]

    for i, flow_pair in flow_enum:
        layer_name = "loss" + str(i + 2)

        flow_scale = final_flow_scale / (2 ** i)

        with tf.variable_scope(layer_name):
            layer_weight = layer_weights[i]
            flow_fw_s, flow_bw_s = flow_pair

            mask_occlusion = params.get('mask_occlusion', '')
            assert mask_occlusion in ['fb', 'disocc', '']

            mask_type = params.get('mask_type', '')
            if mask_type is '':
                mask_type = 'binary'
            assert mask_type in ['linear', 'binary']

            losses = compute_losses(im1_s, im2_s,
                                    flow_fw_s * flow_scale, flow_bw_s * flow_scale, num_layer,
                                    border_mask=mask_s if params.get('border_mask') else None,
                                    mask_occlusion=mask_occlusion,
                                    data_max_distance=layer_patch_distances[i],
                                    mask_type=mask_type)
            layer_loss = 0.0

            for loss in LOSSES:
                weight_name = loss + '_weight'
                if params.get(weight_name):
                    _track_loss(losses[loss], loss)
                    layer_loss += losses[loss] #params[weight_name]
                    combined_losses[loss] += layer_weight * losses[loss]

            '''if i < 1:
                im1_s = downsample(im1_s, 2)
                im2_s = downsample(im2_s, 2)
                mask_s = downsample(mask_s, 2)'''

            im1_s = downsample(im1_s, 2)
            im2_s = downsample(im2_s, 2)
            mask_s = downsample(mask_s, 2)

    regularization_loss = tf.losses.get_regularization_loss()

    loss_weights = dict()

    combined_loss = 0.0
    with tf.variable_scope('trainable_weights'):
        if params.get('motion_weight'):
            LOSSES += ['motion']
            MASK_LOSSES +=['motion_mask']
        for loss in LOSSES:
            loss_weights[loss] = slim.variable(loss + '_weight', dtype=tf.float32, initializer=tf.constant(0.0))
            combined_losses[loss] = tf.exp(-loss_weights[loss]) * combined_losses[loss] + loss_weights[loss]
            combined_loss += combined_losses[loss]

    for loss in MASK_LOSSES:
        weight_name = loss + '_weight'
        if params.get(weight_name):
            combined_losses[loss] = params[weight_name] * combined_losses[loss]
            combined_loss += combined_losses[loss]

    for loss in LOSSES:
        weight_name = loss + '_weight'
        if params.get(weight_name):
            _track_loss(combined_losses[loss], 'loss/' + loss)
            weight = tf.identity(tf.exp(-loss_weights[loss]), name='weight/' + loss)
            tf.add_to_collection('params', weight)

    for loss in MASK_LOSSES:
        weight_name = loss + '_weight'
        if params.get(weight_name):
            _track_loss(combined_losses[loss], 'loss/' + loss)
            weight = tf.identity(params[weight_name], name='weight/' + loss)
            tf.add_to_collection('params', weight)

    final_loss = combined_loss + regularization_loss
    _track_loss(final_loss, 'loss/combined')

    if not return_flow:
        return final_loss

    return final_flow_bw, final_flow_fw, final_flow_bw, im1


def gaussian_kernel(size, mean, std, layers):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel / tf.reduce_sum(gauss_kernel)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    output = gauss_kernel
    for i in range(layers-1):
        output = tf.concat([output, gauss_kernel], axis=2)
    return output


def gaussian_blur(im1, im2, size, mean, std, layers):
    gauss_kernel = gaussian_kernel(size, mean, std, layers) * (2*np.pi) ** 0.5 * std
    im1 = tf.nn.depthwise_conv2d(im1, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    im2 = tf.nn.depthwise_conv2d(im2, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    return [im1, im2]
