import tensorflow as tf
import numpy as np
import math


_correlation_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./build/correlation.so"))


def correlation(input_a, input_b, kernel_size, max_displacement, stride_1, stride_2, padding):
    return _correlation_ops.correlation(input_a,
                                        input_b,
                                        kernel_size,
                                        max_displacement,
                                        stride_1,
                                        stride_2,
                                        padding)


@tf.RegisterGradient("Correlation")
def _correlation_grad(corr_op, gradients):
    kernel_size = corr_op.get_attr("kernel_size")
    max_displacement = corr_op.get_attr("max_displacement")
    stride_1 = corr_op.get_attr("stride_1")
    stride_2 = corr_op.get_attr("stride_2")
    pad = corr_op.get_attr("pad")

    corr_grads = _correlation_ops.correlation_grad(gradients,
                                                   corr_op.inputs[0],
                                                   corr_op.inputs[1],
                                                   kernel_size,
                                                   max_displacement,
                                                   stride_1,
                                                   stride_2,
                                                   pad)

    # Return the gradients with respect to input_a and input_b
    return corr_grads.backprops_a, corr_grads.backprops_b





BATCH_SIZE = 8
HEIGHT = 30
WIDTH = 60
CHANNELS = 3


# Define two feature maps
A = tf.ones((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=tf.float32)
B = tf.convert_to_tensor(np.random.randint(5, size=(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)), dtype=tf.float32)


def corr(fmA, fmB, stride=2, maxdisp=21):
    out = []
    for i in range(-maxdisp + 1, maxdisp, stride): # height
        for j in range(-maxdisp + 1, maxdisp, stride): # width
            padded_a = tf.pad(fmA, [[0,0], [0, abs(i)], [0, abs(j)], [0, 0]])
            padded_b = tf.pad(fmB, [[0, 0], [abs(i), 0], [abs(j), 0], [0, 0]])
            m = padded_a * padded_b

            height_start_idx = 0 if i <= 0 else i
            height_end_idx = height_start_idx + HEIGHT
            width_start_idx = 0 if j <= 0 else j
            width_end_idx = width_start_idx + WIDTH
            cut = m[:, height_start_idx:height_end_idx, width_start_idx:width_end_idx, :]

            final = tf.reduce_sum(cut, 3)
            out.append(final)
    corr = tf.stack(out, 3)
    return corr

	
with(tf.Session()):
	c1 = corr(A, B)
	c2 = correlation(A,B, 1, 20, 1, 2, 20)
	
	print(tf.reduce_sum(c1-c2).eval())
	
	

