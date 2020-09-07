import os
import sys
import shutil

import tensorflow as tf
import numpy as np
import numpy.ma
import png
import timeit
from scipy.ndimage import filters
from PIL import Image, ImageDraw

from e2eflow.core.flow_util import flow_to_color, flow_error_avg, outlier_pct
from e2eflow.core.flow_util import flow_error_image
from e2eflow.util import config_dict
from e2eflow.core.image_warp import image_warp
from e2eflow.kitti.eval_input import KITTIInput
from e2eflow.kitti.data import KITTIData
from e2eflow.core.unsupervised import unsupervised_loss
from e2eflow.core.eval_input import resize_input, resize_output_crop, resize_output, resize_output_flow
from e2eflow.core.train import restore_networks
from e2eflow.ops import forward_warp
from e2eflow.gui import display
from e2eflow.core.losses import DISOCC_THRESH, occlusion, create_outgoing_mask, multi_channels_to_grayscale
from e2eflow.util import convert_input_strings
from ReadData import Matrix, readlabel
from eval.motion_evaluator import evaluate, rotationMatrixToEulerAngles, convert_flow_to_rgb, Euler_angle_to_rot_mat
from eval.tracking_evaluator import motion_eval
from eval.plot import KittiPlotTrajectories
from eval.eval_err import eval_err
from eval.read_label import Read_Label
from eval.plot_motion import KittiPlotMotion


tf.app.flags.DEFINE_string('dataset', 'kitti',
                           'Name of dataset to evaluate on. One of {kitti, sintel, chairs, mdb}.')
tf.app.flags.DEFINE_string('variant', 'grid_map',
                           'Name of variant to evaluate on.')
tf.app.flags.DEFINE_string('ex', '',
                           'Experiment name(s) (can be comma separated list).')
tf.app.flags.DEFINE_string('eval_txt', '08',
                           'Number of evaluate dataset')
tf.app.flags.DEFINE_string('mode', 'estimated',
                           'Choose between real or estimated self movement')

tf.app.flags.DEFINE_integer('num', 100,
                            'Number of examples to evaluate. Set to -1 to evaluate all.')
tf.app.flags.DEFINE_integer('num_vis', -1,
                            'Number of evalutations to visualize. Set to -1 to visualize all.')
tf.app.flags.DEFINE_string('gpu', '0',
                           'GPU device to evaluate on.')

tf.app.flags.DEFINE_boolean('output_visual', True,
                            'Output flow visualization files.')
tf.app.flags.DEFINE_boolean('output_backward', False,
                            'Output backward flow files.')
tf.app.flags.DEFINE_boolean('output_png', False, # TODO finish .flo output
                            'Raw output format to use with output_benchmark.'
                            'Outputs .png flow files if true, output .flo otherwise.')
tf.app.flags.DEFINE_boolean('tracking', False,
                            'Using tracking datasets?')
tf.app.flags.DEFINE_boolean('kitti', False,
                            'For KITTI Benchmark?')
tf.app.flags.DEFINE_boolean('eval_mask', False,
                            'Evaluate motion mask')
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


def write_flow(flow, filename):
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


def _evaluate_experiment(name, input_fn, data_input, matrix_input, layers, track_seq=None, kitti_seq=None):
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

    checkpoint = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
    print(checkpoint)

    #Using tracking?
    output_tracking = FLAGS.tracking
    if output_tracking:
        save_path = '/home/zqhyyl/track/'
        if not os.path.exists(save_path + track_seq[0]):
            os.makedirs(save_path + track_seq[0])

        label_list = readlabel(track_seq[0])
        err_tracking = []
        file_track = open(save_path + "track_err" + track_seq[0] + ".txt", "w")

    with tf.Graph().as_default(): #, tf.device('gpu:' + FLAGS.gpu):
        inputs = input_fn()
        im1, im2, input_shape = inputs[:3]

        height, width, _ = tf.unstack(tf.squeeze(input_shape), num=3, axis=0)
        im1 = resize_input(im1, height, width, layers, resized_h, resized_w)
        im2 = resize_input(im2, height, width, layers, resized_h, resized_w) # TODO adapt train.py

        loss, flow, flow_bw, im1 = unsupervised_loss(
            (im1, im2), checkpoint,
            normalization=data_input.get_normalization(),
            params=params, augment=False, return_flow=True, evaluate=True, eval_mask=FLAGS.eval_mask)

        im1 = resize_output(im1, height, width, layers)
        im2 = resize_output(im2, height, width, layers)
        flow = resize_output_flow(flow, height, width, 2)
        flow_bw = resize_output_flow(flow_bw, height, width, 2)

        flow_fw_int16 = flow_to_int16(flow)
        flow_bw_int16 = flow_to_int16(flow_bw)

        # Evaluate flow with odometry data
        flow_u, flow_v = tf.unstack(flow, axis=3)

        image_slots = [#(loss, 'im1'),
                       (flow_to_color(flow, max_flow=10.0), 'flow'),
                       (multi_channels_to_grayscale(im1, layers), 'gray_img'),
                       (tf.reshape(multi_channels_to_grayscale(im2, layers), [-1]), 'img_array'),
                       (tf.reshape(flow_u, [-1]), 'flow_u_array'),
                       (tf.reshape(flow_v, [-1]), 'flow_v_array')]

        num_ims = len(image_slots)
        image_ops = [t[0] for t in image_slots]
        image_names = [t[1] for t in image_slots]
        all_ops = image_ops + [loss]

        image_lists = []
        sess_config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            odo_save_path = "/home/zqhyyl/"
            if kitti_seq is not None:
                odo_save_path += "kitti_odometry"
                seq = str(kitti_seq[0])
            elif track_seq is not None:
                odo_save_path += "track"
                seq = str(track_seq[0])
            else:
                odo_save_path += "odometry"
                seq = FLAGS.eval_txt
            if not os.path.exists(odo_save_path):
                os.makedirs(odo_save_path)

            exp_out_dir = os.path.join('../out', seq)
            if FLAGS.output_visual:
                if os.path.isdir(exp_out_dir):
                    shutil.rmtree(exp_out_dir)
                os.makedirs(exp_out_dir)
                shutil.copyfile(config_path, os.path.join(exp_out_dir, 'config.ini'))

            if matrix_input is not None:
                R = rotationMatrixToEulerAngles(matrix_input.return_matrix(0)[:, 0:3])[2]
                t = [0.0, 0.0]
                file_err = open(odo_save_path + "/" + seq + "_err.txt", "w")
            else:
                R = 0.0
                t = [0.0, 0.0]
                file_err = None

            file = open(odo_save_path + "/" + seq + "_odo.txt", "w")
            file_kitti = open(odo_save_path + "/" + seq + "_kitti.txt", "w")

            restore_networks(sess, params, ckpt, ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,
                                                   coord=coord)

            # TODO adjust for batch_size > 1 (also need to change image_lists appending)
            if output_tracking:
                max_iter = track_seq[1] if track_seq[1] > 0 else None
            elif FLAGS.kitti:
                max_iter = kitti_seq[1] if kitti_seq[1] > 0 else None
            else:
                max_iter = FLAGS.num if FLAGS.num > 0 else None

            try:
                num_iters = 0
                while not coord.should_stop() and (max_iter is None or num_iters != max_iter):
                    start_sess = timeit.default_timer()
                    all_results = sess.run([flow, flow_bw, flow_fw_int16, flow_bw_int16] + all_ops)
                    stop_sess = timeit.default_timer()
                    all_results = all_results[4:]
                    image_results = all_results[:num_ims]
                    scalar_results = all_results[num_ims:]
                    iterstr = str(num_iters).zfill(6)
                    if FLAGS.output_visual:
                        path_flow = os.path.join(exp_out_dir, iterstr + '_flow.png')
                        im1 = image_results[0] * 255
                        mask = image_results[1] > 0.001
                        im1 = np.multiply(im1, mask)
                        im1[np.where((im1 == [0, 0, 0]).all(axis=-1))] = [255, 255, 255]
                        write_rgb_png(im1, path_flow)
                        if FLAGS.eval_mask:
                            path_flow2 = os.path.join(exp_out_dir, iterstr + '_estflow.png')
                            im1 = (scalar_results[0][0,:,:,:]*255).astype(np.uint8)
                            im1 = np.concatenate([im1, im1, im1], axis=-1)
                            im1 = Image.fromarray(im1)
                            im1.save(path_flow2)
                    if num_iters < FLAGS.num_vis:
                        image_lists.append(image_results)
                    if num_iters > 0:
                        sys.stdout.write('\r')

                    start_est = timeit.default_timer()
                    R, t = evaluate(file, file_err, file_kitti, R, t,  matrix_input, num_iters, image_results[1:5], FLAGS.mode)

                    if output_tracking:
                        labels_crt = Read_Label(label_list, num_iters, image_results[1])
                        labels_nxt = Read_Label(label_list, num_iters+1, image_results[1])
                        motion_eval(labels_crt, labels_nxt, image_results[1:5], num_iters+3, err_tracking, track_seq)

                    stop_est = timeit.default_timer()

                    num_iters += 1
                    sys.stdout.write("-- evaluating '{}': {}/{}"
                                     .format(name, num_iters, max_iter))
                    sys.stdout.flush()

                    print()
                    print('Motion Estimation Time: ', stop_est - start_est)
                    print('Inference time per Image: ', stop_sess - start_sess)
                    #print('Loss:', scalar_results[0])
                    print()

            except tf.errors.OutOfRangeError:
                pass

            rot_mat = Euler_angle_to_rot_mat([0.0, R, 0.0]).flatten()
            # write image and odometry
            angle_text = ''
            for text in rot_mat:
                angle_text += str(text) + ' '
            #file_kitti.write(angle_text + "0 " + str(-t[1]) + " " + str(t[0]) + "\n")

            file_kitti.close()
            file.close()
            coord.request_stop()
            coord.join(threads)

    if output_tracking:
        err = [0.0, 0.0]
        err_0 = []
        err_1 = []
        for errors in err_tracking:
            err[0] += abs(errors[0])
            err[1] += abs(errors[1])
            err_0.append(errors[0])
            err_1.append(errors[1])

        file_track.write("Total_tracked_objects: " + str(len(err_tracking)) + " Total_err: " + str(err) +
                         " average_error: " + str([errs / float(len(err_tracking)) for errs in err]) +
                         " maximum_error_X: " + str(max(err_0)) + " minimum_error_X: " + str(min(err_0)) +
                         " maximum_error_Y: " + str(max(err_1)) + " minimum_error_Y: " + str(min(err_1)))
        file_track.close()


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    for name in FLAGS.ex.split(','):
        config_path = '/home/zqhyyl/Desktop/UnFlow/log/ex/' + name + '/config.ini'
    default_config = config_dict(config_path)
    dirs = '/home/zqhyyl/Desktop/UnFlow'
    kconfig = default_config['train_kitti']
    layers = kconfig.get('layers').split(', ')
    num_layers = 0
    for layer in layers:
        if layer == 'rgb_cartesian':
            num_layers = num_layers + 3
        else:
            num_layers = num_layers + 1
    height = kconfig.get('height')
    width = kconfig.get('width')
    print(layers)

    # Input odometry data
    try:
        matrix_dir = '/home/zqhyyl/pcl_data/gridmap_train/odo_gt/poses/' + FLAGS.eval_txt + '.txt'
        matrix_input = Matrix(dir=matrix_dir)
        matrix_input.return_matrix(0)
    except FileNotFoundError:
        matrix_input = None

    odo_save_path = "/home/zqhyyl/"
    if FLAGS.tracking:
        print("-- evaluating on all tracking datasets")
        odo_save_path += "odometry"
        for eval_seq in [['0009', 801], ['0010', 292]]:
            data = KITTIData(dirs, development=True)
            data_input = KITTIInput(data, eval_seq[0], batch_size=1, normalize=False,
                                    dims=(height, width), layers=layers, num_layers=num_layers)
            input_fn = getattr(data_input, 'input_' + FLAGS.variant)

            for name in FLAGS.ex.split(','):
                _evaluate_experiment(name, input_fn, data_input, None, num_layers, track_seq=eval_seq)
    elif FLAGS.kitti:
        print("-- evaluating for KITTI Benchmark")
        odo_save_path += "kitti_odometry"
        for eval_seq in [['11', 920], ['12', 1059], ['13', 3279], ['14', 629], ['15', 1900]]:
            data = KITTIData(dirs, development=True)
            data_input = KITTIInput(data, eval_seq[0], batch_size=1, normalize=False,
                                    dims=(height, width), layers=layers, num_layers=num_layers)
            input_fn = getattr(data_input, 'input_' + FLAGS.variant)

            for name in FLAGS.ex.split(','):
                _evaluate_experiment(name, input_fn, data_input, matrix_input, num_layers, kitti_seq=eval_seq)
                KittiPlotTrajectories(str(eval_seq[0]), eval_seq[1], odo_save_path, matrix_input=matrix_input)
                KittiPlotMotion(str(eval_seq[0]), eval_seq[1], odo_save_path)
    else:
        print("-- evaluating: on {} pairs from {}/{}"
              .format(FLAGS.num, FLAGS.dataset, FLAGS.variant))
        odo_save_path += "odometry"

        data = KITTIData(dirs, development=True)
        data_input = KITTIInput(data, str(FLAGS.eval_txt), batch_size=1, normalize=False,
                                dims=(height, width), layers=layers, num_layers=num_layers)
        input_fn = getattr(data_input, 'input_' + FLAGS.variant)

        for name in FLAGS.ex.split(','):
            _evaluate_experiment(name, input_fn, data_input, matrix_input, num_layers)
            KittiPlotTrajectories(FLAGS.eval_txt, FLAGS.num, odo_save_path, matrix_input=matrix_input)

        if matrix_input is not None:
            KittiPlotMotion(FLAGS.eval_txt, FLAGS.num, odo_save_path)


if __name__ == '__main__':
    tf.app.run()
