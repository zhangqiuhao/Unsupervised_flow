import tensorflow as tf


def decode_csv(line):
    rec_defaults = [[""],[""],[""],[""]]

    vec = tf.decode_csv(line, rec_defaults, field_delim=',')
    image_str_ch0 = tf.read_file(vec[0])
    image_ch0 = (tf.cast(tf.image.decode_png(image_str_ch0, channels=3, dtype=tf.uint16), dtype=tf.float32) - 32768) / 200
    image_str_ch1 = tf.read_file(vec[1])
    image_ch1 = (tf.cast(tf.image.decode_png(image_str_ch1, channels=3, dtype=tf.uint16), dtype=tf.float32) - 32768) / 200

    image_str_gt_ch0 = tf.read_file(vec[2])
    image_gt = (tf.cast(tf.image.decode_png(image_str_gt_ch0, channels=3, dtype=tf.uint16), dtype=tf.float32) - 32768) / 2000

    return tf.reshape(tf.stack([image_ch0, image_ch1], axis=2), [64,1440,6]), tf.reshape(image_gt, [64,1440,3])


def input_fn(batch_size, path):
    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(lambda line: decode_csv(line))

    dataset = dataset.shuffle(150)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def predict_input_fn(vector):
    dataset = tf.data.Dataset.from_tensor_slices(vector)

    dataset = dataset.map(lambda line: decode_csv(line))

    dataset = dataset.batch(1)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def main():
    return


if __name__ == "__main__":
    main()