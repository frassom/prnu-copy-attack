import numpy as np
from tqdm import tqdm

import prnu

__rng = np.random.default_rng(42)


def __random_int(rng: np.random.Generator, min_val: int, max_val: int, available: int) -> int:
    """
    Generate a random number between min_val and max_val given the available space, taking care of
    leaving enough space for the next iteration.
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
        shape of the image to be divided
    block_shape : tuple of int, optional
        the block shape (im_shape[i] == N * block_shape[i])

    Returns
    -------
    list of (slice,slice)
    """
    assert(im_shape[0] % block_shape[0] == 0)
    assert(im_shape[1] % block_shape[1] == 0)

    blocks = list()

    for i0 in range(im_shape[0] // block_shape[0]):
        for i1 in range(im_shape[1] // block_shape[1]):
            blocks.append(np.s_[i0 * block_shape[0]:i0 * block_shape[0] + block_shape[0],
                                i1 * block_shape[1]:i1 * block_shape[1] + block_shape[1]])
    return blocks


def gen_blocks_rnd(im_shape, min_block_shape=(16, 16), max_block_shape=(30, 30), rng=None):
    """
    Generate a list of pair of slices representing the blocks the images are divided into,
    each block size is randomized

    Parameters
    ----------
    im_shape : tuple of int
        shape of the image to be divided
    min_block_shape : tuple of int, optional
        minimum blocks shape
    max_block_shape : tuple of int, optional
        maximum blocks shape
    rng : numpy.random.Generator, optional
        numpy random generator if None the default is used

    Returns
    -------
    blocks : list of (slice, slice)
    """

    if not rng:
        rng = __rng

    blocks = list()

    i0 = 0
    while i0 < im_shape[0]:
        height = __random_int(
            rng, min_block_shape[0], max_block_shape[0], im_shape[0] - i0)

        i1 = 0
        while i1 < im_shape[1]:
            width = __random_int(
                rng, min_block_shape[1], max_block_shape[1], im_shape[1] - i1)

            blocks.append(np.s_[i0:i0+height, i1:i1+width])
            i1 += width
        i0 += height

    return blocks


def extract_prnu_var(imgs, blocks, lum_range, r, R=None, rng=None, **kwargs):
    """
    Extract prnu noise from a list of images of same size, using one of the variance-based methods

    Parameters
    ----------
    imgs : list of numpy.ndarray
        images of shape (h, w, ch) and type np.uint8 used to extract prnu
    blocks : list of (slice, slice) 
        division in block of images for prnu extraction, see gen_block[_rnd](...)
    lum_range : tuple of int
        (t_low, t_up) range of luminance value for which a block is accepted
    r : int
        number of block to use for noise extraction
    R : int, optional
        number of lowest variance blocks to retain in ranVar attacks for randomized selection of r blocks
    rng : numpy.random.Generator, optional
        the numpy random generator to use, if None the default is used
    **kwargs : dict, optional
        additional parameters to pass to extract_multiple_aligned() in prnu-python

    Returns
    -------
    prnu : numpy.ndarray
    """

    if not rng:
        rng = __rng

    K = np.empty(imgs[0].shape[0:2])

    # Compute variance for each block
    for b in tqdm(blocks):
        var = list()
        for im in imgs:
            im_gray = prnu.rgb2gray(im)

            mean = np.mean(im_gray[b])
            if lum_range[0] <= mean <= lum_range[1]:
                var.append([np.var(im_gray[b]), im[b]])

        var.sort(key=lambda a: a[0])
        if not R:
            var = np.array(var[:r], copy=False, dtype=object)
        else:
            var = rng.choice(var[:R], r, replace=False)

        K[b] = prnu.extract_multiple_aligned(var.T[1], **kwargs)

    return K
