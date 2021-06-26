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


def generate_blocks(im_shape: tuple, randomize: bool, rng: np.random.Generator = None, **kwargs) -> list:
    """
    Generate a list of pair of slices representing the blocks the images are divided into.
    Depending on the attack type blocks all have the same size or are random in size.

    :param im_shape: shape of the images
    :param randomize: if true the block sizes will be randomized
    :param rng: numpy random generator used if randomize == True

    kwargs
    ------
    :param block_shape: shape of the block if randomize == False
    :param min_block_shape: minimum shape of blocks if randomize == True
    :param max_block_shape: maximum shape of blocks if randomize == True

    :return: list of (slice,slice)
    """
    blocks = list()

    if not rng:
        rng = __rng

    if randomize:
        block_shape = kwargs.get("block_shape", (16, 16))

        assert(im_shape[0] % block_shape[0] == 0)
        assert(im_shape[1] % block_shape[1] == 0)

        for i0 in range(im_shape[0] // block_shape[0]):
            for i1 in range(im_shape[1] // block_shape[1]):
                blocks.append(np.s_[i0 * block_shape[0]:i0 * block_shape[0] + block_shape[0],
                                    i1 * block_shape[1]:i1 * block_shape[1] + block_shape[1]])
    else:
        min_block_shape = kwargs.get("min_block_shape", (16, 16))
        max_block_shape = kwargs.get("max_block_shape", (30, 30))

        i0 = 0
        while i0 < im_shape[0]:
            height = __random_int(
                rng, min_block_shape[0], max_block_shape[0], im_shape[0] - i0)

            i1 = 0
            while i1 < im_shape[1]:
                width = __random_int(
                    rng, min_block_shape[1], max_block_shape[1], im_shape[1] - i1)

                blocks.append(np.s_[i0:i0+height,
                                    i1:i1+width])
                i1 += width
            i0 += height

    return blocks


def extract_prnu_var(imgs: list, blocks: list, r: int, lum_range: tuple,
                     R: int = None, rng: np.random.Generator = None, **kwargs) -> np.ndarray:
    """
    Extract prnu noise from a list of images of same size, using one of the variance-based methods

    :param imgs: list of np.ndarray of shape (h, w, ch) and type np.uint8 used to extract prnu
    :param blocks: list of (slice, slice) in which images are subdivided for prnu extraction
    :param r: number of block to use for noise extraction
    :param lum_range: (t_low, t_up) range of luminance value for which a block is accepted
    :param R: number of lowest variance blocks to retain in <ranVar-*> attacks for randomized selection of r blocks later
    :param rng: the numpy random generator to use in <randVar-*> attacks
    :param kwargs: additional parameter to pass to extract_multiple_aligned() in prnu-python

    :return: prnu noise
    """

    K = np.empty((imgs[0].shape[0], imgs[0].shape[1]))

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
            if not rng:
                rng = __rng
            var = rng.choice(var[:R], r, replace=False)

        K[b] = prnu.extract_multiple_aligned(var.T[1], **kwargs)

    return K
