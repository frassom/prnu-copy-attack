# Copyright (c) 2021 Marco Frassineti
# Copyright (c) 2018 Nicolò Bonettini, Paolo Bestagini, Luca Bondi
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

import numpy as np


def cut_center(array, shape):
    """
    Cut a multidimensional array at its center, according to shape.

    Parameters
    ----------
    array : numpy.ndarray
        Multidimensional array.
    shape : tuple of int
        Tuple of the same length as array.ndim, resulting shape.

    Returns
    -------
    numpy.ndarray
        Center cut of the input multidimensional array.
    """

    if not (array.ndim == len(shape)):
        raise ValueError('array.ndim must be equal to len(sizes)')

    array = array.copy()

    for axis in range(array.ndim):
        axis_target_size = shape[axis]
        axis_original_size = array.shape[axis]
        if axis_target_size > axis_original_size:
            raise ValueError(
                f'can\'t have target size {axis_target_size} for axis {axis}'
                f'with original size {axis_original_size}')
        elif axis_target_size < axis_original_size:
            axis_start_idx = (axis_original_size - axis_target_size) // 2
            axis_end_idx = axis_start_idx + axis_target_size
            indices = np.arange(axis_start_idx, axis_end_idx)
            array = np.take(array, indices, axis)

    return array


def rgb2gray(im, dtype=np.float32):
    """
    RGB to gray as from Binghamton toolbox.

    Parameters
    ----------
    im : numpy.ndarray
        Multidimensional array.
    dtype : data-type
        Data type of the returned array.

    Returns
    -------
    im_gray : numpy.ndarray
        Grayscale version of input im.
    """

    rgb2gray_vector = \
        np.asarray([0.29893602, 0.58704307, 0.11402090], dtype=np.float32)

    if im.ndim == 2:
        im_gray = np.array(im, copy=True, dtype=dtype)
    elif im.shape[2] == 1:
        im_gray = np.array(im[:, :, 0], copy=True, dtype=dtype)
    elif im.shape[2] == 3:
        w, h = im.shape[:2]
        im = np.reshape(im, (w * h, 3))
        im_gray = np.dot(im, rgb2gray_vector)
        im_gray.shape = (w, h)
    else:
        raise ValueError('input image must have 1 or 3 channels')

    if im_gray.dtype == dtype:
        return im_gray
    else:
        return im_gray.astype(dtype)


def get_rng(rnd):
    """Return a numpy.random.Generator from the value of rnd."""

    if rnd is None:
        return np.random.default_rng()
    elif isinstance(rnd, int):
        return np.random.default_rng(rnd)
    elif isinstance(rnd, np.random.Generator):
        return rnd

    raise ValueError(
        "rnd must be None, int or numpy Generator instance")


def _random_int(rng, min_val, max_val, available):
    """
    Generate a random number between min_val and max_val given the available space,
    taking care of leaving enough space for the next iteration.

    Parameters
    ----------
    rng : numpy.random.Generator
    min_val, max_val : int
        Range of the value.
    available : int
        Value used to cumpute how much "space" is left, if there is not enough space
        this value is returned.

    Returns
    -------
    int
        Random integer.
    """

    if available - min_val >= max_val:
        return rng.integers(min_val, max_val + 1)
    elif available - min_val >= min_val:
        return rng.integers(min_val, available - min_val + 1)
    else:
        return available


def gen_blocks(im_shape, block_shape=(16, 16)):
    """
    Generate a list of pair of slices representing the blocks the images are divided into.

    Parameters
    ----------
    im_shape : tuple of int
        Dhape of the image to be divided.
    block_shape : tuple of int, optional
        The block shape (im_shape[i] == N * block_shape[i]).

    Returns
    -------
    blocks : list of (slice,slice)
    """

    assert(im_shape[0] % block_shape[0] == 0)
    assert(im_shape[1] % block_shape[1] == 0)

    blocks = list()

    for i0 in range(im_shape[0] // block_shape[0]):
        for i1 in range(im_shape[1] // block_shape[1]):
            blocks.append(np.s_[i0 * block_shape[0]:i0 * block_shape[0] + block_shape[0],
                                i1 * block_shape[1]:i1 * block_shape[1] + block_shape[1]])
    return blocks


def gen_blocks_rnd(im_shape, min_block_shape=(16, 16), max_block_shape=(40, 40), rnd=None):
    """
    Generate a list of pair of slices representing the blocks the images are divided into,
    as in gen_blocks, each row has randomized height and each block has randomized width.

    Parameters
    ----------
    im_shape : tuple of int
        Shape of the image to be divided.
    min_block_shape : tuple of int, optional
        Minimum blocks shape.
    max_block_shape : tuple of int, optional
        Maximum blocks shape.
    rng : numpy.random.Generator, optional
        Numpy random generator if None the default is used.

    Returns
    -------
    blocks : list of (slice, slice)
    """

    rng = get_rng(rnd)

    blocks = list()

    i0 = 0
    while i0 < im_shape[0]:
        height = _random_int(
            rng, min_block_shape[0], max_block_shape[0], im_shape[0] - i0)

        i1 = 0
        while i1 < im_shape[1]:
            width = _random_int(
                rng, min_block_shape[1], max_block_shape[1], im_shape[1] - i1)

            blocks.append(np.s_[i0:i0+height, i1:i1+width])
            i1 += width
        i0 += height

    return blocks
