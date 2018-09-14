import tensorflow as tf
import sys
from tools.PlotDirectionalField import directionalFieldTF


def model_fn(features, labels, mode, params):
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = labels[:,:,:,0:2]

    if params['RESIZE_WIDTH'] != 0:
        _, h, _, _ = features.shape
        features = tf.image.resize_images(features, [h, params['RESIZE_WIDTH']])
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = tf.image.resize_images(labels, [h, params['RESIZE_WIDTH']])

    if params['CROP_HEIGHT'] != 0:
        features = features[:,:params['CROP_HEIGHT'], :, :]
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = labels[:,:params['CROP_HEIGHT'], :, :]

    x = features

    res_dict = {}
    for layer, _ in enumerate(params['CONVOLUTION_LAYERS']['SIZES']):

        if params['CONVOLUTION_LAYERS']['RESIDUAL'][layer].startswith('get'):
            res_dict[params['CONVOLUTION_LAYERS']['RESIDUAL'][layer]] = x

        x = tf.layers.conv2d(x, filters=params['CONVOLUTION_LAYERS']['FILTERS'][layer],
                              kernel_size=params['CONVOLUTION_LAYERS']['SIZES'][layer],
                              dilation_rate=params['CONVOLUTION_LAYERS']['DILATION'][layer],
                              padding='same', name='conv'+str(layer))

        if params['CONVOLUTION_LAYERS']['POOLING'][layer] is not 0:
            x = tf.layers.max_pooling2d(x, params['CONVOLUTION_LAYERS']['POOLING'][layer], params['CONVOLUTION_LAYERS']['POOLING'][layer], padding='same')

        if params['CONVOLUTION_LAYERS']['RESIDUAL'][layer].startswith('add'):
            x = tf.add(x,res_dict[params['CONVOLUTION_LAYERS']['RESIDUAL'][layer].replace('add','get')],name=params['CONVOLUTION_LAYERS']['RESIDUAL'][layer])

        if params['CONVOLUTION_LAYERS']['RESIDUAL'][layer].startswith('stack'):
            x = tf.concat([x,res_dict[params['CONVOLUTION_LAYERS']['RESIDUAL'][layer].replace('stack','get')]],axis=3,name=params['CONVOLUTION_LAYERS']['RESIDUAL'][layer])

        if params['CONVOLUTION_LAYERS']['ACTIVATION'][layer] == 'relu':
            x = tf.nn.relu(x, name='relu'+str(layer))
        if params['CONVOLUTION_LAYERS']['BATCHNORM'][layer]:
            x = tf.layers.batch_normalization(x, training=training, momentum=0.9)

    x = x[:, :, :, 0:2]

    # for prediction only:
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"predictions": x})

    # calculate loss:
    mask = tf.stack([features[:, :, :, 0], features[:, :, :, 0]], axis=3)
    weight_matrix = tf.where(tf.abs(mask) < 0.001, tf.ones_like(mask), tf.ones_like(mask)*params['LOSS_WEIGHT_OCCUPIED'])

    if params['LOSS'] == 'l1':
        loss = tf.losses.absolute_difference(labels, x, weight_matrix)
    if params['LOSS'] == 'l2':
        loss = tf.losses.mean_squared_error(labels, x, weight_matrix)

    # output end point error as evaluation metric
    firstlayer = features[:, :, :, 0]
    bool_mask = tf.where(tf.abs(firstlayer) < 0.001, tf.zeros_like(firstlayer, dtype=tf.bool), tf.ones_like(firstlayer, dtype=tf.bool))
    epe = tf.reduce_mean(tf.boolean_mask(tf.norm(labels - x, ord='euclidean', axis=3), bool_mask))
    tf.summary.scalar('epe', epe)

    # image output
    tf.summary.image('output', tf.expand_dims(directionalFieldTF(x[0,:,:,0],x[0,:,:,1]),0))
    tf.summary.image('groundtruth', tf.expand_dims(directionalFieldTF(labels[0,:,:,0],labels[0,:,:,1]),0))

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
