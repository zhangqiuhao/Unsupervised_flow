#!/usr/bin/env python3
import tensorflow as tf
import argparse
import yaml
import importlib
from hooks.Hooks import SaveTrainableParamsCount


def main(args):
    import os

    # load config file:
    with open(args.parameters, 'r') as stream:
        try:
            cfg = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    # import model & data function
    model = importlib.import_module(cfg['MODEL_FN'])
    data = importlib.import_module(cfg['INPUT_FN'])

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

    nn = tf.estimator.Estimator(model_fn=model.model_fn, params=cfg, config=config)

    if not args.inference:
        # first run to load pretrained weights
        if cfg['FIRSTRUN']:
            nn.train(input_fn=lambda:data.input_fn(cfg['BATCH_SIZE'], cfg['TRAIN_DATA_LOCATION']),
                     hooks=[SaveTrainableParamsCount(cfg['MODEL_DIR'])], steps=1)

        cfg.update({'FIRSTRUN': False})
        nn = tf.estimator.Estimator(model_fn=model.model_fn, params=cfg, config=config)

        for x in range(cfg['NUM_EPOCHS']):
            nn.train(input_fn=lambda:data.input_fn(cfg['BATCH_SIZE'], cfg['TRAIN_DATA_LOCATION']))
            nn.evaluate(input_fn=lambda:data.input_fn(cfg['BATCH_SIZE'], cfg['EVAL_DATA_LOCATION']), steps=int(cfg['EVAL_EXAMPLES']/cfg['BATCH_SIZE']))
    else:
        import numpy as np

        if args.output != '':
            if not os.path.exists(args.output):
                os.makedirs(args.output)

        with open(cfg['EVAL_DATA_LOCATION'], 'r') as f:
            slice_data = [line.strip() for line in f]

        predictions = nn.predict(input_fn=lambda:data.predict_input_fn(slice_data))

        if args.output == '':
            # test inference time:
            import time
            millis = lambda: int(round(time.time() * 1000))

            starttime = 0
            for i,p in enumerate(predictions):
                print(i)
                if i == 10:
                    starttime = millis()
                if i == 1010:
                    timediff = millis() - starttime
                    print(timediff/1000)
                    break

        else:

            if not os.path.exists(args.output):
                os.makedirs(args.output)

            for i,p in enumerate(predictions):
                prediction = p['predictions']
                data = slice_data[i].split(',')

                # write to output dir:
                print('Writing estimation',i+1,'of',len(slice_data))
                np.save(args.output + '/' + str(i).zfill(3) + '_estimation', np.array(prediction))
    return


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser(description='Neural Network')
    parser.add_argument('-p', '--parameters', help='Yaml parameter file to be used')
    parser.add_argument('-i', '--inference', help='Set this for prediction, omit for learning.', action='store_true')
    parser.add_argument('-o', '--output', help='If set, this is the output folder for prediction.', default='')

    main(parser.parse_args())

