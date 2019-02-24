"""
An implementation of a multilinear interpolator in tensorflow.
"""
import tensorflow as tf


def _resize_by_axis_trilinear(images, size_0, size_1, ax):
    """
    Resize image bilinearly to [size_0, size_1] except axis ax.
        :param image: a tensor 4-D with shape 
                        [batch, d0, d1, d2, channels]
        :param size_0: size 0
        :param size_1: size 1
        :param ax: axis to exclude from the interpolation
    """
    resized_list = []

    # unstack the image in 2d cases
    unstack_list = tf.unstack(images, axis = ax)
    for i in unstack_list:
        # resize bilinearly
        resized_list.append(tf.image.resize_bilinear(i, [size_0, size_1]))
    stack_img = tf.stack(resized_list, axis=ax)

    return stack_img


def resize_trilinear(images, size):
    """
    Resize images to size using trilinear interpolation.
        :param images: A tensor 5-D with shape 
                        [batch, d0, d1, d2, channels]
        :param size: A 1-D int32 Tensor of 3 elements: new_d0, new_d1,
                        new_d2. The new size for the images.
    """
    assert size.shape[0] == 3
    resized = _resize_by_axis_trilinear(images, size[0], size[1], 2)
    resized = _resize_by_axis_trilinear(resized, size[0], size[2], 1)
    return resized


def _resize_by_axis_tetralinear(images, size_0, size_1, size_2, ax):
    """
    Resize image trilinearly to [size_0, size_1, size_2] except axis ax.
        :param image: a tensor 6-D with shape 
                        [batch, d0, d1, d2, d3, channels]
        :param size_0: size 0
        :param size_1: size 1
        :param size_2: size 2
        :param ax: axis to exclude from the interpolation
    """
    resized_list = []

    # unstack the image in 3d cases
    unstack_list = tf.unstack(images, axis = ax)
    for i in unstack_list:
        # resize trilinearly
        new_size = tf.constant([size_0, size_1, size_2])
        resized_list.append(resize_trilinear(i, new_size))
    stack_img = tf.stack(resized_list, axis=ax)

    return stack_img


def resize_tetralinear(images, size):
    """
    Resize images to size using tetralinear interpolation.
        :param images: A tensor 6-D with shape 
                        [batch, d0, d1, d2, d3, channels]
        :param size: A 1-D int32 Tensor of 4 elements: new_d0, new_d1,
                        new_d2, new_d3. The new size for the images.
    """
    assert size.shape[0] == 4
    resized = _resize_by_axis_tetralinear(images, size[0], size[1], size[2], 4)
    resized = _resize_by_axis_tetralinear(resized, size[0], size[1], size[3], 3)
    return resized


def resize_multilinear_tf(images, size):
    """
    Resize images to size using multilinear interpolation.
        :param images: A tensor with shape 
                        [batch, d0, ..., dn, channels]
        :param size: A 1-D int32 Tensor. The new size for the images.
    """
    if size.shape[0] == 2:
        resized = tf.image.resize_bilinear(images, size)
    elif size.shape[0] == 3:
        resized = resize_trilinear(images, size)
    elif size.shape[0] == 4:
        resized = resize_tetralinear(images, size)
    else:
        raise NotImplementedError('resize_multilinear_tf: dimensions \
                                    higuer than 4 are not supported.')
    return resized
