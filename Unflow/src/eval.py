import os
import sys
import shutil

import tensorflow as tf
import numpy as np
import numpy.ma
import png
from PIL import Image
import math

from e2eflow.core.flow_util import flow_to_color, flow_error_avg, outlier_pct
from e2eflow.core.flow_util import flow_error_image
from e2eflow.util import config_dict
from e2eflow.core.image_warp import image_warp
from e2eflow.kitti.eval_input import KITTIInput
from e2eflow.kitti.data import KITTIData
from e2eflow.core.unsupervised import unsupervised_loss
from e2eflow.core.input import resize_input, resize_output_crop, resize_output, resize_output_flow
from e2eflow.core.train import restore_networks
from e2eflow.ops import forward_warp
from e2eflow.gui import display
from e2eflow.core.losses import DISOCC_THRESH, occlusion, create_outgoing_mask
from e2eflow.util import convert_input_strings
from Matrix import Matrix


tf.app.flags.DEFINE_string('dataset', 'kitti',
                            'Name of dataset to evaluate on. One of {kitti, sintel, chairs, mdb}.')
tf.app.flags.DEFINE_string('variant', 'grid_map',
                           'Name of variant to evaluate on.')
tf.app.flags.DEFINE_string('ex', '',
                           'Experiment name(s) (can be comma separated list).')
tf.app.flags.DEFINE_string('eval_txt', '08',
                           'Number of evaluate dataset')
tf.app.flags.DEFINE_integer('num', 4000,
                            'Number of examples to evaluate. Set to -1 to evaluate all.')
tf.app.flags.DEFINE_integer('num_vis', -1,
                            'Number of evalutations to visualize. Set to -1 to visualize all.')
tf.app.flags.DEFINE_string('gpu', '0',
                           'GPU device to evaluate on.')
tf.app.flags.DEFINE_boolean('output_benchmark', False,
                            'Output raw flow files.')
tf.app.flags.DEFINE_boolean('output_visual', True,
                            'Output flow visualization files.')
tf.app.flags.DEFINE_boolean('output_backward', False,
                            'Output backward flow files.')
tf.app.flags.DEFINE_boolean('output_png', False, # TODO finish .flo output
                            'Raw output format to use with output_benchmark.'
                            'Outputs .png flow files if true, output .flo otherwise.')
FLAGS = tf.app.flags.FLAGS


NUM_EXAMPLES_PER_PAGE = 4


def write_rgb_png(z, path, bitdepth=8):
    z = z[0, :, :, :]
    with open(path, 'wb') as f:
        writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=bitdepth)
        z2list = z.reshape(-1, z.shape[1]*z.shape[2]).tolist()
        writer.write(f, z2list)


def flow_to_int16(flow):
    _, h, w, _ = tf.unstack(tf.shape(flow))
    u, v = tf.unstack(flow, num=2, axis=3)
    r = tf.cast(tf.maximum(0.0, tf.minimum(u * 64.0 + 32768.0, 65535.0)), tf.uint16)
    g = tf.cast(tf.maximum(0.0, tf.minimum(v * 64.0 + 32768.0, 65535.0)), tf.uint16)
    b = tf.ones([1, h, w], tf.uint16)
    return tf.stack([r, g, b], axis=3)


def write_flo(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    flow = flow[0, :, :, :]
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()


def _evaluate_experiment(name, input_fn, data_input, matrix_input):
    normalize_fn = data_input._normalize_image
    resized_h = data_input.dims[0]
    resized_w = data_input.dims[1]

    current_config = config_dict('../config.ini')
    exp_dir = os.path.join(current_config['dirs']['log'], 'ex', name)
    config_path = os.path.join(exp_dir, 'config.ini')
    if not os.path.isfile(config_path):
        config_path = '../config.ini'
    if not os.path.isdir(exp_dir) or not tf.train.get_checkpoint_state(exp_dir):
        exp_dir = os.path.join(current_config['dirs']['checkpoints'], name)
    config = config_dict(config_path)
    params = config['train']
    convert_input_strings(params, config_dict('../config.ini')['dirs'])
    dataset_params_name = 'train_' + FLAGS.dataset
    if dataset_params_name in config:
        params.update(config[dataset_params_name])
    ckpt = tf.train.get_checkpoint_state(exp_dir)
    if not ckpt:
        raise RuntimeError("Error: experiment must contain a checkpoint")
    ckpt_path = exp_dir + "/" + os.path.basename(ckpt.model_checkpoint_path)

    with tf.Graph().as_default(): #, tf.device('gpu:' + FLAGS.gpu):
        inputs = input_fn()
        im1, im2, input_shape = inputs[:3]

        height, width, _ = tf.unstack(tf.squeeze(input_shape), num=3, axis=0)
        im1 = resize_input(im1, height, width, resized_h, resized_w)
        im2 = resize_input(im2, height, width, resized_h, resized_w) # TODO adapt train.py

        _, flow, flow_bw = unsupervised_loss(
            (im1, im2),
            normalization=data_input.get_normalization(),
            params=params, augment=False, return_flow=True)

        im1 = resize_output(im1, height, width, 3)
        im2 = resize_output(im2, height, width, 3)
        flow = resize_output_flow(flow, height, width, 2)
        flow_bw = resize_output_flow(flow_bw, height, width, 2)

        flow_fw_int16 = flow_to_int16(flow)
        flow_bw_int16 = flow_to_int16(flow_bw)

        im1_pred = image_warp(im2, flow)
        im1_diff = tf.abs(im1 - im1_pred)
        #im2_diff = tf.abs(im1 - im2)

        #flow_bw_warped = image_warp(flow_bw, flow)
        #div = divergence(flow_occ)
        #div_bw = divergence(flow_bw)

        # Evaluate flow with odometry data

        occ_pred = 1 - (1 - occlusion(flow, flow_bw)[0])
        def_pred = 1 - (1 - occlusion(flow, flow_bw)[1])
        disocc_pred = forward_warp(flow_bw) < DISOCC_THRESH
        disocc_fw_pred = forward_warp(flow) < DISOCC_THRESH

        flow_u, flow_v = tf.unstack(flow, axis=3)

        image_slots = [((im2 * 0.5 + im1_pred * 0.5) / 255, 'overlay'),
                       (im1_pred / 255, 'img1_pred'),
                       #(im1_diff / 255, 'brightness error'),
                       #(im1 / 255, 'first image', 1, 0),
                       #(im2 / 255, 'second image', 1, 0),
                       #(im2_diff / 255, '|first - second|', 1, 2),
                       (flow_to_color(flow), 'flow'),
                       (tf.image.rgb_to_grayscale(im2), 'gray_img'),
                       (tf.reshape(tf.image.rgb_to_grayscale(im2), [-1]), 'img_array'),
                       (tf.reshape(flow_u, [-1]), 'flow_u_array'),
                       (tf.reshape(flow_v, [-1]), 'flow_v_array')
                       #(flow_to_color(flow_bw), 'flow bw prediction'),
                       #(tf.image.rgb_to_grayscale(im1_diff) > 20, 'diff'),
                       #(div, 'div'),
                       #(div < -2, 'neg div'),
                       #(div > 5, 'pos div'),
                       #  (blue: correct, red: wrong, dark: occluded)
        ]

        num_ims = len(image_slots)
        image_ops = [t[0] for t in image_slots]
        image_names = [t[1] for t in image_slots]
        all_ops = image_ops

        image_lists = []
        sess_config = tf.ConfigProto(allow_soft_placement=True)

        exp_out_dir = os.path.join('../out', name)
        if FLAGS.output_visual or FLAGS.output_benchmark:
            if os.path.isdir(exp_out_dir):
                shutil.rmtree(exp_out_dir)
            os.makedirs(exp_out_dir)
            shutil.copyfile(config_path, os.path.join(exp_out_dir, 'config.ini'))

        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            R = 0.0
            t = matrix_input.return_matrix(2)[:,3]
            file = open("/home/zhang/odo.txt", "w")

            restore_networks(sess, params, ckpt, ckpt_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,
                                                   coord=coord)

            # TODO adjust for batch_size > 1 (also need to change image_lists appending)
            max_iter = FLAGS.num if FLAGS.num > 0 else None

            try:
                num_iters = 0
                while not coord.should_stop() and (max_iter is None or num_iters != max_iter):
                    all_results = sess.run([flow, flow_bw, flow_fw_int16, flow_bw_int16] + all_ops)
                    flow_fw_res, flow_bw_res, flow_fw_int16_res, flow_bw_int16_res = all_results[:4]
                    all_results = all_results[4:]
                    image_results = all_results[:num_ims]
                    scalar_results = all_results[num_ims:]
                    iterstr = str(num_iters).zfill(6)
                    if FLAGS.output_visual:
                        path_flow = os.path.join(exp_out_dir, iterstr + '_flow.png')
                        path_overlay = os.path.join(exp_out_dir, iterstr + '_overlay.png')
                        path_fw_overlay = os.path.join(exp_out_dir, iterstr + '_fw.png')
                        #write_rgb_png(image_results[0] * 255, path_overlay)
                        #write_rgb_png(image_results[1] * 255, path_fw_overlay)
                        write_rgb_png(image_results[2] * 255, path_flow)
                    if FLAGS.output_benchmark:
                        path_fw = os.path.join(exp_out_dir, iterstr)
                        if FLAGS.output_png:
                            write_rgb_png(flow_fw_int16_res, path_fw  + '_10.png', bitdepth=16)
                        else:
                            write_flo(flow_fw_res, path_fw + '_10.flo')
                        if FLAGS.output_backward:
                            path_fw = os.path.join(exp_out_dir, iterstr + '_01.png')
                            write_rgb_png(flow_bw_int16_res, path_bw, bitdepth=16)
                    if num_iters < FLAGS.num_vis:
                        image_lists.append(image_results)
                    if num_iters > 0:
                        sys.stdout.write('\r')

                    R,t = _evaluate(file, R, t, matrix_input, num_iters, image_results[3], image_results[4], image_results[5], image_results[6])

                    num_iters += 1
                    sys.stdout.write("-- evaluating '{}': {}/{}"
                                     .format(name, num_iters, max_iter))
                    sys.stdout.flush()
                    print()
            except tf.errors.OutOfRangeError:
                pass

            file.close()
            coord.request_stop()
            coord.join(threads)

    return image_lists, image_names

def _evaluate(file, old_R, old_t, matrix_input, num_iters, img_gray, img_array, flow_u_array, flow_v_array):
    height = img_gray.shape[1]
    width = img_gray.shape[2]

    matrix_im1 = []
    matrix_im2 = []
    count = 0

    output = np.zeros([height, width], 'uint8')
    output_flow = np.zeros([height, width], 'uint8')

    for u in range(height):
        for v in range(width):
            num_pixel = v + u * width
            if img_array[num_pixel] > 0.001 and abs(flow_u_array[num_pixel]) > 0.001 and abs(flow_v_array[num_pixel]) > 0.001:
                output_u = u - flow_v_array[num_pixel]
                output_v = v - flow_u_array[num_pixel]
                if  0 <= int(output_u) < height and 0 <= int(output_v) < width:

                    #output_flow[int(output_u), int(output_v)] = img_array[num_pixel]
                    #output[u, v] = img_array[num_pixel]

                    matrix_im1.append([output_u, output_v, 0])
                    matrix_im2.append([u, v, 0])
                    count += 1

    #img = Image.fromarray(output)
    #img_flow = Image.fromarray(output_flow)

    #img.save('../../out/' + str(num_iters) + '_output.jpeg')
    #img_flow.save('../../out/' + str(num_iters) + '_output_flow.jpeg')

    matrix_im1 = np.transpose(np.asarray(matrix_im1))
    matrix_im2 = np.transpose(np.asarray(matrix_im2))

    print("Number of points: ",count)

    R, c, t = ralign(matrix_im1, matrix_im2)
    #print(matrix_real[:,0:2].shape)
    #angles_real = rotationMatrixToEulerAngles(matrix_real[:,:3])
    #R = np.dot(old_R, R)
    #t = np.transpose(np.add(-t * 0.1, np.transpose(old_t)))

    t_x = t[0] * np.cos(old_R) - t[1] * np.sin(old_R)
    t_y = t[0] * np.sin(old_R) + t[1] * np.cos(old_R)

    R = old_R - rotationMatrixToEulerAngles(R)[2]
    #print("angles is: ", angles_is, "\nreal angles:  ", angles_real)
    t = [(t_x * 0.1+old_t[0]).item(0) , (t_y * 0.1+old_t[1]).item(0), (t[2]*0.1+old_t[2]).item(0)]
    file.write("0 0 0 " + str(t[0]) + " 0 0 0 " + str(t[1]) + " 0 0 0 " + str(t[2]) + "\n")

    print("Rotation angle: ", R, "Translation: ", t)
    #matrix_is = np.column_stack((R,t))
    #diff = np.subtract(matrix_real, matrix_is)

    #print("\nRotation matrix=\n", R, "\nScaling coefficient=", c, "\nTranslation vector=", t)
    #print("Error_matrix=\n", diff)

    return R, t

def ralign(X,Y):
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.rank(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.det(Sxy) < 0 ):
            S[m, m] = -1;
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R,c,t

    R = np.dot( np.dot(U, S ), V.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R,c,t


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    print("-- evaluating: on {} pairs from {}/{}"
          .format(FLAGS.num, FLAGS.dataset, FLAGS.variant))

    default_config = config_dict()
    dirs = default_config['dirs']

    # Input odometry data
    matrix_dir = dirs['odo_dirs'] + FLAGS.eval_txt + '.txt'
    matrix_input = Matrix(dir = matrix_dir)

    data = KITTIData(dirs['data'], development=True)
    data_input = KITTIInput(data, batch_size=1, normalize=False,
                                 dims=(640,640))
    input_fn = getattr(data_input, 'input_' + FLAGS.variant)

    results = []
    for name in FLAGS.ex.split(','):
        result, image_names = _evaluate_experiment(name, input_fn, data_input, matrix_input)
        results.append(result)

    #display(results, image_names)

if __name__ == '__main__':
    tf.app.run()
