import tensorflow as tf
import numpy as np

_rgb2ycbcr_kernel = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=np.float32,
)

_ycbcr2rgb_kernel = np.array(
    [
        [1, 0.0, 1.402],
        [1, -0.344136, -0.714136],
        [1, 1.772, 0.0],
    ],
    dtype=np.float32,
)


_offset = tf.constant([0, 128, 128], dtype=tf.float32, name="offset")


@tf.function
def rgb2ycbcr(image: tf.Tensor) -> tf.Tensor:
    value = tf.cast(image, tf.float32)
    kernel = tf.constant(_rgb2ycbcr_kernel, dtype=tf.float32, name="kernel")
    value = tf.tensordot(value, tf.transpose(kernel), axes=[(-1,), (0,)])
    value = tf.add(value, _offset)
    return tf.clip_by_value(value, tf.constant([0], dtype=tf.float32), tf.constant([255], dtype=tf.float32))

@tf.function
def ycbcr2rgb(image: tf.Tensor) -> tf.Tensor:
    value = tf.cast(image, tf.float32)
    value = tf.subtract(value, _offset)
    kernel = tf.constant(_ycbcr2rgb_kernel, dtype=tf.float32, name="kernel")
    value = tf.tensordot(value, tf.transpose(kernel), axes=[(-1,), (0,)])
    return tf.clip_by_value(value, tf.constant([0], dtype=tf.float32), tf.constant([255], dtype=tf.float32))
