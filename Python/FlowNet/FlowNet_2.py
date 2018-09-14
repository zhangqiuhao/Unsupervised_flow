import tensorflow as tf
import argparse
import yaml
from tensorflow.python.ops.init_ops import Initializer
import numpy as np
from tools.DataInputCylindrical import predict_input_fn, input_fn
from hooks.Hooks import SaveTrainableParamsCount
from tools.PlotDirectionalField import directionalFieldTF
from flownet_ops.op_correlation import correlation
from flownet_ops.op_flow_warp import flow_warp
from math import pi


class LoadInitializer(Initializer):

    def __init__(self, dtype=tf.float32, name='', trainable=True):
        self.dtype = tf.as_dtype(dtype)
        self.filename = '/home/klein/U/extracted_weights/flownet_2/' + name + '.npy'

        if not trainable:  # load own pretrained fixed weights
            self.filename = '/home/klein/U/extracted_weights/flownet_csssd_kitti/' + name + '.npy'

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        print('Loading ', self.filename)
        data = np.load(self.filename)
        tensor = tf.reshape(tf.convert_to_tensor(data, dtype=dtype), shape)

        return tensor

    def get_config(self):
        return {"dtype": self.dtype.name}
    

def conv(a, filters, name, kernel_size=1, strides=1, activation=None, fr=False, reuse=None, trainable=True):
    if activation is None: activation = tf.nn.leaky_relu
    if activation == 'None': activation = None
    if fr:
        return tf.layers.conv2d(a, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name=name, activation=activation, reuse=reuse, kernel_initializer=LoadInitializer(name=name+'_weights', trainable=trainable),  bias_initializer=LoadInitializer(name=name+'_biases', trainable=trainable), trainable=trainable)
    else:
        return tf.layers.conv2d(a, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name=name, activation=activation, reuse=reuse, trainable=trainable)


def deconv(a, filters, name, kernel_size=1, strides=1, activation=None, fr=False, trainable=True):
    if activation is None: activation = tf.nn.leaky_relu
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

    #####################################
    # Flownet C with fixed weights:
    #####################################
    with tf.variable_scope('FlowNet_C'):

        image_a = features[:,:,:,0:3]
        image_b = features[:,:,:,3:6]

        conv_a_1 = conv(image_a, filters=64, kernel_size=7, strides=2, fr=firstrun, trainable=False, name='FlowNet_C_conv1')
        conv_a_2 = conv(conv_a_1, filters=128, kernel_size=5, strides=2, fr=firstrun, trainable=False, name='FlowNet_C_conv2')
        conv_a_3 = conv(conv_a_2, filters=256, kernel_size=5, strides=2, fr=firstrun, trainable=False, name='FlowNet_C_conv3')

        conv_b_1 = conv(image_b, filters=64, kernel_size=7, strides=2, fr=firstrun, trainable=False, activation=tf.nn.relu, reuse=True, name='FlowNet_C_conv1')
        conv_b_2 = conv(conv_b_1, filters=128, kernel_size=5, strides=2, fr=firstrun, trainable=False, activation=tf.nn.relu, reuse=True, name='FlowNet_C_conv2')
        conv_b_3 = conv(conv_b_2, filters=256, kernel_size=5, strides=2, fr=firstrun, trainable=False, activation=tf.nn.relu, reuse=True, name='FlowNet_C_conv3')

        cc = correlation(conv_a_3, conv_b_3, 1, 20, 1, 2, 20)
        cc_relu = tf.nn.relu(cc)
        netA_conv = conv(conv_a_3, filters=32, kernel_size=1, fr=firstrun, trainable=False, name='FlowNet_C_conv_redir')
        net = tf.concat([netA_conv, cc_relu], axis=3)

        conv3_1 = conv(net, filters=256, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_C_conv3_1')

        conv4 = conv(conv3_1, filters=512, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_C_conv4')
        conv4_1 = conv(conv4, filters=512, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_C_conv4_1')

        conv5 = conv(conv4_1, filters=512, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_C_conv5')
        conv5_1 = conv(conv5, filters=512, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_C_conv5_1')

        conv6 = conv(conv5_1, filters=1024, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_C_conv6')
        conv6_1 = conv(conv6, filters=1024, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_C_conv6_1')

        # refinement
        predict_6 = conv(conv6_1, filters=2, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_C_predict_flow6')
        deconv5 = deconv(conv6_1, filters=512, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_C_deconv5')
        upsample_6to5 = deconv(predict_6, filters=2, kernel_size=4, strides=2, activation=None, fr=firstrun, trainable=False, name='FlowNet_C_upsample_flow6to5')
        concat5 = tf.concat([conv5_1, deconv5, upsample_6to5], axis=3, name='FlowNet_C_concat_1')

        predict_5 = conv(concat5, filters=2, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_C_predict_flow5')
        deconv4 = deconv(concat5, filters=256, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_C_deconv4')
        upsample_5to4 = deconv(predict_5, filters=2, kernel_size=4, strides=2, activation=None, fr=firstrun, trainable=False, name='FlowNet_C_upsample_flow5to4')
        concat4 = tf.concat([conv4_1, deconv4, upsample_5to4], axis=3, name='FlowNet_C_concat_2')

        predict_4 = conv(concat4, filters=2, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_C_predict_flow4')
        deconv3 = deconv(concat4, filters=128, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_C_deconv3')
        upsample_4to3 = deconv(predict_4, filters=2, kernel_size=4, strides=2, activation=None, fr=firstrun, trainable=False, name='FlowNet_C_upsample_flow4to3')
        concat3 = tf.concat([conv3_1, deconv3, upsample_4to3], axis=3, name='FlowNet_C_concat_3')

        predict_3 = conv(concat3, filters=2, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_C_predict_flow3')
        deconv2 = deconv(concat3, filters=64, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_C_deconv2')
        upsample_3to2 = deconv(predict_3, filters=2, kernel_size=4, strides=2, activation=None, fr=firstrun, trainable=False, name='FlowNet_C_upsample_flow3to2')
        concat2 = tf.concat([conv_a_2, deconv2, upsample_3to2], axis=3, name='FlowNet_C_concat_4')

        predict_2 = conv(concat2, filters=3, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_C_predict_2')  # no pretrained weights available because we want 3d-flow

        x = tf.image.resize_bilinear(predict_2, [64, 1024], align_corners=True)

    # END OF FROZEN FLOWNET_C
    # tf.summary.image('c_out', tf.expand_dims(directionalFieldTF(x[0,:,:,0],x[0,:,:,1],1.8),0))

    # calculating pixel-flow for warping:
    startpixel = (1 - tf.atan2(features[:,:,:,1], features[:,:,:,0]) / pi) * 512
    endpixel = (1 - tf.atan2(features[:,:,:,1] + x[:,:,:,1], features[:,:,:,0] + x[:,:,:,0]) / pi) * 512
    pixelflow = tf.stack([tf.zeros_like(startpixel), endpixel-startpixel], axis=3)

    # Perform flow warping (to move image B closer to image A based on flow prediction)
    warped = flow_warp(features[:,:,:,3:6], pixelflow)

    # Compute brightness error: sqrt(sum (input_a - warped)^2 over channels)
    brightness_error = features[:,:,:,0:3] - warped
    brightness_error = tf.square(brightness_error)
    brightness_error = tf.reduce_sum(brightness_error, keep_dims=True, axis=3)
    brightness_error = tf.sqrt(brightness_error)

    # stacking all new layers:
    x = tf.concat([features, warped, x[:,:,:,0:2], brightness_error], axis=3)
    
    #####################################
    # Flownet S with fixed weights:
    #####################################
    with tf.variable_scope('FlowNet_S_fixed'):

        x = conv(x, filters=64, kernel_size=7, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_conv1')
        x = conv(x, filters=128, kernel_size=5, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_conv2')
        x_conv2 = x
        x = conv(x, filters=256, kernel_size=5, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_conv3')
        x = conv(x, filters=256, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_S_conv3_1')
        x_conv3_1 = x
        x = conv(x, filters=512, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_conv4')
        x = conv(x, filters=512, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_S_conv4_1')
        x_conv4_1 = x
        x = conv(x, filters=512, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_conv5')
        x = conv(x, filters=512, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_S_conv5_1')
        x_conv5_1 = x
        x = conv(x, filters=1024, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_conv6')
        x = conv(x, filters=1024, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_S_conv6_1')

        # refinement
        predict_6 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_S_predict_flow6')
        x = deconv(x, filters=512, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_deconv5')
        upsample_6to5 = deconv(predict_6, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_upsample_flow6to5')
        x = tf.concat([x_conv5_1, x, upsample_6to5], axis=3, name='concat_1')

        predict_5 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_S_predict_flow5')
        x = deconv(x, filters=256, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_deconv4')
        upsample_5to4 = deconv(predict_5, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_upsample_flow5to4')
        x = tf.concat([x_conv4_1, x, upsample_5to4], axis=3, name='concat_2')

        predict_4 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_S_predict_flow4')
        x = deconv(x, filters=128, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_deconv3')
        upsample_4to3 = deconv(predict_4, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_upsample_flow4to3')
        x = tf.concat([x_conv3_1, x, upsample_4to3], axis=3, name='concat_3')

        predict_3 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_S_predict_flow3')
        x = deconv(x, filters=64, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_deconv2')
        upsample_3to2 = deconv(predict_3, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S_upsample_flow3to2')
        x = tf.concat([x_conv2, x, upsample_3to2], axis=3, name='concat_4')

        predict_2 = conv(x, filters=3, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_S_predict_2')
        x = tf.image.resize_bilinear(predict_2, [64, 1024], align_corners=True)

    # END OF FROZEN FLOWNET_S
    # tf.summary.image('cs_out', tf.expand_dims(directionalFieldTF(x[0,:,:,0],x[0,:,:,1],1.8),0))

    # calculating pixel-flow for warping:
    startpixel = (1 - tf.atan2(features[:,:,:,1], features[:,:,:,0]) / pi) * 512
    endpixel = (1 - tf.atan2(features[:,:,:,1] + x[:,:,:,1], features[:,:,:,0] + x[:,:,:,0]) / pi) * 512
    pixelflow = tf.stack([tf.zeros_like(startpixel), endpixel-startpixel], axis=3)

    # Perform flow warping (to move image B closer to image A based on flow prediction)
    warped = flow_warp(features[:,:,:,3:6], pixelflow)

    # Compute brightness error: sqrt(sum (input_a - warped)^2 over channels)
    brightness_error = features[:,:,:,0:3] - warped
    brightness_error = tf.square(brightness_error)
    brightness_error = tf.reduce_sum(brightness_error, keep_dims=True, axis=3)
    brightness_error = tf.sqrt(brightness_error)

    # stacking all new layers:
    x = tf.concat([features, warped, x[:,:,:,0:2], brightness_error], axis=3)

    #####################################
    # Flownet S with Fixed weights:
    #####################################
    with tf.variable_scope('FlowNet_S2_fixed'):

        x = conv(x, filters=64, kernel_size=7, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_conv1')
        x = conv(x, filters=128, kernel_size=5, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_conv2')
        x_conv2 = x
        x = conv(x, filters=256, kernel_size=5, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_conv3')
        x = conv(x, filters=256, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_S2_conv3_1')
        x_conv3_1 = x
        x = conv(x, filters=512, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_conv4')
        x = conv(x, filters=512, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_S2_conv4_1')
        x_conv4_1 = x
        x = conv(x, filters=512, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_conv5')
        x = conv(x, filters=512, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_S2_conv5_1')
        x_conv5_1 = x
        x = conv(x, filters=1024, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_conv6')
        x = conv(x, filters=1024, kernel_size=3, strides=1, fr=firstrun, trainable=False, name='FlowNet_S2_conv6_1')

        # refinement
        predict_6 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_S2_predict_flow6')
        x = deconv(x, filters=512, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_deconv5')
        upsample_6to5 = deconv(predict_6, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_upsample_flow6to5')
        x = tf.concat([x_conv5_1, x, upsample_6to5], axis=3, name='concat_1')

        predict_5 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_S2_predict_flow5')
        x = deconv(x, filters=256, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_deconv4')
        upsample_5to4 = deconv(predict_5, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_upsample_flow5to4')
        x = tf.concat([x_conv4_1, x, upsample_5to4], axis=3, name='concat_2')

        predict_4 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_S2_predict_flow4')
        x = deconv(x, filters=128, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_deconv3')
        upsample_4to3 = deconv(predict_4, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_upsample_flow4to3')
        x = tf.concat([x_conv3_1, x, upsample_4to3], axis=3, name='concat_3')

        predict_3 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_S2_predict_flow3')
        x = deconv(x, filters=64, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_deconv2')
        upsample_3to2 = deconv(predict_3, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_S2_upsample_flow3to2')
        x = tf.concat([x_conv2, x, upsample_3to2], axis=3, name='concat_4')

        predict_2 = conv(x, filters=3, kernel_size=3, activation='None', fr=firstrun, trainable=False, name='FlowNet_S2_predict_2')
        x = tf.image.resize_bilinear(predict_2, [64, 1024], align_corners=True)

    # end of fixed flownet_s
    tf.summary.image('css_out', tf.expand_dims(directionalFieldTF(x[0,:,:,0],x[0,:,:,1],1.8),0))
    css_result = x

    ####################################################################################################################
    ####################################################################################################################

    #####################################
    # Flownet SD with Fixed weights:
    #####################################

    with tf.variable_scope('FlowNet_SD_fixed'):
        x = features

        x = conv(x, filters=64, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_SD_conv0')
        x = conv(x, filters=64, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_conv1')
        x = conv(x, filters=128, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_SD_conv1_1')
        x = conv(x, filters=128, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_conv2')
        x_conv2 = x
        x = conv(x, filters=128, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_SD_conv2_1')
        x = conv(x, filters=256, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_conv3')
        x = conv(x, filters=256, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_SD_conv3_1')
        x_conv3_1 = x
        x = conv(x, filters=512, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_conv4')
        x = conv(x, filters=512, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_SD_conv4_1')
        x_conv4_1 = x
        x = conv(x, filters=512, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_conv5')
        x = conv(x, filters=512, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_SD_conv5_1')
        x_conv5_1 = x
        x = conv(x, filters=1024, kernel_size=3, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_conv6')
        x = conv(x, filters=1024, kernel_size=3, fr=firstrun, trainable=False, name='FlowNet_SD_conv6_1')

        # refinement
        predict_6 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_SD_predict_flow6')
        x = deconv(x, filters=512, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_deconv5')
        upsample_6to5 = deconv(predict_6, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_upsample_flow6to5')
        x = tf.concat([x_conv5_1, x, upsample_6to5], axis=3, name='FlowNet_SD_concat_1')
        x = conv(x, filters=512, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_SD_interconv5')

        predict_5 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_SD_predict_flow5')
        x = deconv(x, filters=256, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_deconv4')
        upsample_5to4 = deconv(predict_5, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_upsample_flow5to4')
        x = tf.concat([x_conv4_1, x, upsample_5to4], axis=3, name='FlowNet_SD_concat_2')
        x = conv(x, filters=256, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_SD_interconv4')

        predict_4 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_SD_predict_flow4')
        x = deconv(x, filters=128, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_deconv3')
        upsample_4to3 = deconv(predict_4, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_upsample_flow4to3')
        x = tf.concat([x_conv3_1, x, upsample_4to3], axis=3, name='FlowNet_SD_concat_3')
        x = conv(x, filters=128, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_SD_interconv3')

        predict_3 = conv(x, filters=2, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_SD_predict_flow3')
        x = deconv(x, filters=64, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_deconv2')
        upsample_3to2 = deconv(predict_3, filters=2, kernel_size=4, strides=2, fr=firstrun, trainable=False, name='FlowNet_SD_upsample_flow3to2')
        x = tf.concat([x_conv2, x, upsample_3to2], axis=3, name='FlowNet_SD_concat_4')
        x = conv(x, filters=64, kernel_size=3, activation=None, fr=firstrun, trainable=False, name='FlowNet_SD_interconv2')

        predict_2 = conv(x, filters=3, kernel_size=3, activation='None', fr=firstrun, trainable=False, name='FlowNet_SD_predict_2')
        x = tf.image.resize_bilinear(predict_2, [64, 1024], align_corners=True)

    tf.summary.image('sd_out', tf.expand_dims(directionalFieldTF(x[0,:,:,0],x[0,:,:,1],1.8),0))
    sd_result = x
    ####################################################################################################################
    ####################################################################################################################


    def ChannelNorm(tensor):
        sq = tf.square(tensor)
        r_sum = tf.reduce_sum(sq, keep_dims=True, axis=3)
        return tf.sqrt(r_sum)

    def PixelFlow(features_, x_):
        # calculating pixel-flow for warping:
        startpixel_ = (1 - tf.atan2(features_[:,:,:,1], features_[:,:,:,0]) / pi) * 512
        endpixel_ = (1 - tf.atan2(features_[:,:,:,1] + x_[:,:,:,1], features_[:,:,:,0] + x_[:,:,:,0]) / pi) * 512
        return tf.stack([tf.zeros_like(startpixel_), endpixel_-startpixel_], axis=3)

    sd_flow_norm = ChannelNorm(sd_result[:,:,:,0:2])
    css_flow_norm = ChannelNorm(css_result[:,:,:,0:2])

    pixelflow_sd = PixelFlow(features, sd_result)
    flow_warp_sd = flow_warp(features[:,:,:,3:6], pixelflow_sd)
    img_diff_sd = features[:,:,:,0:3] - flow_warp_sd
    img_diff_sd_norm = ChannelNorm(img_diff_sd)

    pixelflow_css = PixelFlow(features, css_result)
    flow_warp_css = flow_warp(features[:,:,:,3:6], pixelflow_css)
    img_diff_css = features[:,:,:,0:3] - flow_warp_css
    img_diff_css_norm = ChannelNorm(img_diff_css)

    input_to_fusion = tf.concat([features[:,:,:,0:3],
                                 sd_result[:,:,:,0:2],
                                 css_result[:,:,:,0:2],
                                 sd_flow_norm,
                                 css_flow_norm,
                                 img_diff_sd_norm,
                                 img_diff_css_norm], axis=3)

    #####################################
    # Trainable Fusion network:
    #####################################

    with tf.variable_scope('Fusion_Network'):
        fuse_conv0 = conv(input_to_fusion, filters=64, kernel_size=3, fr=firstrun, name='fuse_conv0')

        fuse_conv1 = conv(fuse_conv0, filters=64, kernel_size=3, strides=2, fr=firstrun, name='fuse_conv1')
        fuse_conv1_1 = conv(fuse_conv1, filters=128, kernel_size=3, fr=firstrun, name='fuse_conv1_1')

        fuse_conv2 = conv(fuse_conv1_1, filters=128, kernel_size=3, strides=2, fr=firstrun, name='fuse_conv2')
        fuse_conv2_1 = conv(fuse_conv2, filters=128, kernel_size=3, fr=firstrun, name='fuse_conv2_1')

        predict_flow2 = conv(fuse_conv2_1, filters=2, kernel_size=3, activation=None, fr=firstrun, name='predict_flow2')
        fuse_deconv1 = deconv(fuse_conv2_1, filters=32, kernel_size=4, strides=2, fr=firstrun, name='fuse_deconv1')
        fuse_upsample_flow2to1 = deconv(predict_flow2, filters=2, kernel_size=4, strides=2, activation=None, fr=firstrun, name='fuse_upsample_flow2to1')
        concat1 = tf.concat([fuse_conv1_1, fuse_deconv1, fuse_upsample_flow2to1], axis=3)
        fuse_interconv1 = conv(concat1, filters=32, kernel_size=3, activation=None, fr=firstrun, name='fuse_interconv1')

        predict_flow1 = conv(fuse_interconv1, filters=2, kernel_size=3, activation=None, fr=firstrun, name='predict_flow1')
        fuse_deconv0 = deconv(concat1, filters=16, kernel_size=4, strides=2, fr=firstrun, name='fuse_deconv0')
        fuse_upsample_flow1to0 = deconv(predict_flow1, filters=2, kernel_size=4, strides=2, activation=None, fr=firstrun, name='fuse_upsample_flow1to0')
        concat0 = tf.concat([fuse_conv0, fuse_deconv0, fuse_upsample_flow1to0], axis=3)
        fuse_interconv0 = conv(concat0, filters=16, kernel_size=3, activation=None, fr=firstrun, name='fuse_interconv0')

        predict_flow0 = tf.layers.conv2d(fuse_interconv0, filters=3, kernel_size=3, activation=None, name='predict_flow0')
        x = tf.image.resize_bilinear(predict_flow0, [64, 1024], align_corners=True)

    # end trainable fusion network


    # for prediction only:
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"predictions": x})

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
    tf.summary.image('output', tf.expand_dims(directionalFieldTF(x[0,:,:,0],x[0,:,:,1],1.8),0))
    tf.summary.image('groundtruth', tf.expand_dims(directionalFieldTF(labels[0,:,:,0],labels[0,:,:,1],1.8),0))

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


def main(args):
    import os

    # load config file:
    with open(args.parameters, 'r') as stream:
        try:
            cfg = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg['GPU_FRACTION'])
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.estimator.RunConfig()
    config = config.replace(session_config=session_config)
    config = config.replace(model_dir=cfg['MODEL_DIR'])
    config = config.replace(save_summary_steps=cfg['SUMMARY_STEPS'])
    config = config.replace(save_checkpoints_secs=cfg['CHECKPOINTS_SECS'])

    if os.path.exists(cfg['MODEL_DIR']):
        cfg.update({'FIRSTRUN': False})
    else:
        cfg.update({'FIRSTRUN': True})

    nn = tf.estimator.Estimator(model_fn=model_fn, params=cfg, config=config)

    if not args.inference:
        # first run to load pretrained weights
        if cfg['FIRSTRUN']:
            nn.train(input_fn=lambda: input_fn(cfg['BATCH_SIZE'], cfg['TRAIN_DATA_LOCATION']),
                     hooks=[SaveTrainableParamsCount(cfg['MODEL_DIR'])], steps=1)

        cfg.update({'FIRSTRUN': False})
        nn = tf.estimator.Estimator(model_fn=model_fn, params=cfg, config=config)

        for x in range(cfg['NUM_EPOCHS']):
            nn.train(input_fn=lambda:input_fn(cfg['BATCH_SIZE'], cfg['TRAIN_DATA_LOCATION']))
            nn.evaluate(input_fn=lambda:input_fn(cfg['BATCH_SIZE'], cfg['EVAL_DATA_LOCATION']), steps=int(cfg['EVAL_EXAMPLES']/cfg['BATCH_SIZE']))
    else:
        import os

        if args.output != '':
            if not os.path.exists(args.output):
                os.makedirs(args.output)

        with open(cfg['EVAL_DATA_LOCATION'], 'r') as f:
            slice_data = [line.strip() for line in f]

        predictions = nn.predict(input_fn=lambda: predict_input_fn(slice_data))
        for i,p in enumerate(predictions):
            prediction = p['predictions']
            data = slice_data[i].split(',')
            if args.output == '':
                print(prediction)
                input()
            else:
                # write to output dir:
                print('Writing estimation',i+1,'of',len(slice_data))
                np.save(args.output + '/' + str(i).zfill(3) + '_estimation', np.array(prediction))

    return


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser(description='Neural Network')
    parser.add_argument('-p', '--parameters', help='Yaml parameter file to be used')
    parser.add_argument('-i', '--inference', help='Set this for prediciton, omit for learning.', action='store_true')
    parser.add_argument('-o', '--output', help='If set, this is the output folder for prediction.', default='')

    main(parser.parse_args())
