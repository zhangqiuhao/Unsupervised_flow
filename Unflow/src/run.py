import os
import copy

import tensorflow as tf
from tensorflow.python.client import device_lib

from e2eflow.core.train import Trainer
from e2eflow.experiment import Experiment
from e2eflow.util import convert_input_strings

from e2eflow.kitti.input import KITTIInput
from e2eflow.kitti.data import KITTIData


tf.app.flags.DEFINE_string('ex', 'default',
                           'Name of the experiment.'
                           'If the experiment folder already exists in the log dir, '
                           'training will be continued from the latest checkpoint.')
tf.app.flags.DEFINE_boolean('debug', False,
                            'Enable image summaries and disable checkpoint writing for debugging.')
tf.app.flags.DEFINE_boolean('ow', False,
                            'Overwrites a previous experiment with the same name (if present)'
                            'instead of attempting to continue from its latest checkpoint.')
FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    experiment = Experiment(
        name=FLAGS.ex,
        overwrite=FLAGS.ow)
    dirs = experiment.config['dirs']
    run_config = experiment.config['run']

    gpu_list_param = run_config['gpu_list']

    if isinstance(gpu_list_param, int):
        gpu_list = [gpu_list_param]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list_param)
    else:
        gpu_list = list(range(len(gpu_list_param.split(','))))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_param
    gpu_batch_size = int(run_config['batch_size'] / max(len(gpu_list), 1))
    devices = ['/gpu:' + str(gpu_num) for gpu_num in gpu_list]

    train_dataset = run_config.get('dataset', 'kitti')

    kconfig = copy.deepcopy(experiment.config['train'])
    kconfig.update(experiment.config['train_kitti'])
    convert_input_strings(kconfig, dirs)
    kiters = kconfig.get('num_iters', 0)
    layers = kconfig.get('layers').split(', ')
    mask_layers = kconfig.get('mask_layers')
    if mask_layers is not None:
        mask_layers = kconfig.get('mask_layers').split(', ')
    num_layers = 0
    for layer in layers:
        if layer == 'rgb_cartesian':
            num_layers = num_layers + 3
        else:
            num_layers = num_layers + 1
    print(layers)
    print(mask_layers)

    kdata = KITTIData(data_dir=dirs['data'],
                      fast_dir=dirs.get('fast'),
                      stat_log_dir=None,
                      development=run_config['development'])

    if train_dataset == 'kitti':
        kinput = KITTIInput(data=kdata,
                            batch_size=gpu_batch_size,
                            normalize=False,
                            skipped_frames=True,
                            dims=(kconfig['height'], kconfig['width']),
                            layers=layers,
                            num_layers=num_layers,
                            mask_layers=mask_layers)
        tr = Trainer(
              lambda shift: kinput.input_raw(swap_images=False,
                                             center_crop=True,
                                             shift=shift * run_config['batch_size']),
              params=kconfig,
              normalization=kinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, kiters)

    else:
      raise ValueError(
          "Invalid dataset. Dataset must be "
          "{kitti}")

    if not FLAGS.debug:
        experiment.conclude()


if __name__ == '__main__':
    tf.app.run()
