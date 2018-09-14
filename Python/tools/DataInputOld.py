import tensorflow as tf


def decode_csv(line, size):
    rec_defaults = [[""],[""],[""],[""],[0.0],[0.0],[0.0]]
    vec = tf.decode_csv(line, rec_defaults, field_delim=',')
    image_str_ch0 = tf.read_file(vec[0])
    image_ch0 = 255.0 - tf.cast(tf.image.decode_png(image_str_ch0, channels=1), dtype=tf.float32)
    image_str_ch1 = tf.read_file(vec[1])
    image_ch1 = 255.0 - tf.cast(tf.image.decode_png(image_str_ch1, channels=1), dtype=tf.float32)

    image_str_gt_ch0 = tf.read_file(vec[2])
    image_gt_ch0 = tf.cast(tf.image.decode_png(image_str_gt_ch0, channels=1), dtype=tf.float32)
    image_gt_ch0 -= 128.0
    image_str_gt_ch1 = tf.read_file(vec[3])
    image_gt_ch1 = tf.cast(tf.image.decode_png(image_str_gt_ch1, channels=1), dtype=tf.float32)
    image_gt_ch1 -= 128.0

    return tf.reshape(tf.stack([image_ch0, image_ch1], axis=2), [size,size,2]), tf.reshape(tf.stack([image_gt_ch0, image_gt_ch1], axis=2), [size,size,2])


def input_fn(batch_size, path, size=1200):
    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(lambda line: decode_csv(line, size))

    dataset = dataset.shuffle(150)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def predict_input_fn(vector, size=1200):
    dataset = tf.data.Dataset.from_tensor_slices(vector)

    dataset = dataset.map(lambda line: decode_csv(line, size))

    dataset = dataset.batch(1)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def main():
    # Test method

    sess = tf.Session()
    nex = input_fn(1, "/home/klein/U/gridmapsFlow/eval/shuffled.csv", size=1200)
    #nex = predict_input_fn(['/home/klein/U/gridmaps/eval/kitti00/0_euclidean_prob_200_400_103735.png,/home/klein/U/gridmaps/eval/kitti00/0_euclidean_prob_200_400_207338.png,0,0,0','/home/klein/U/gridmaps/eval/kitti00/0_euclidean_prob_200_600_103735.png,/home/klein/U/gridmaps/eval/kitti00/0_euclidean_prob_200_600_207338.png,0,0,0'])

    nex = nex[0]

    nex = tf.stack([nex[:,:,:,0], nex[:,:,:,0]], axis=3)
    nex = tf.where(tf.abs(nex) < 0.001, tf.ones_like(nex), tf.ones_like(nex)*10)

    input = sess.run(nex)
    print(input)

    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(input[0,:,:,0])
    plt.subplot(122)
    plt.imshow(input[0,:,:,1])
    plt.show()


if __name__ == "__main__":
    main()
