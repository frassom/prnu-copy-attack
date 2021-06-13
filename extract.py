import numpy as np

import prnu


def _random_int(rng, min_val: int, max_val: int, available: int) -> int:
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


def generate_blocks(im_shape: tuple, rand_type: str, rng, **kwargs) -> list:
    """
    Generate a list of pair of slices representing the blocks the images are divided into.
    Depending on the attack type blocks all have the same size or are random in size.

    :param im_shape: shape of the images
    :param rand_type: type of attack, same as in extract_prnu_randvar(...)

    :return: list of slices [(slice,slice),...]
    """
    blocks = list()

    if rand_type in [None, "A"]:
        block_shape = kwargs.get("block_shape", (16, 16))

        assert(im_shape[0] % block_shape[0] == 0)
        assert(im_shape[1] % block_shape[1] == 0)

        for i0 in range(im_shape[0] // block_shape[0]):
            for i1 in range(im_shape[1] // block_shape[1]):
                blocks.append(np.s_[i0 * block_shape[0]:i0 * block_shape[0] + block_shape[0],
                                    i1 * block_shape[1]:i1 * block_shape[1] + block_shape[1]])

    elif rand_type == "B":
        min_block_shape = kwargs.get("min_block_shape", (16, 16))
        max_block_shape = kwargs.get("max_block_shape", (30, 30))

        i0 = 0
        while i0 < im_shape[0]:
            height = _random_int(
                rng, min_block_shape[0], max_block_shape[0], im_shape[0] - i0)

            i1 = 0
            while i1 < im_shape[1]:
                width = _random_int(
                    rng, min_block_shape[1], max_block_shape[1], im_shape[1] - i1)

                blocks.append(np.s_[i0:i0+height,
                                    i1:i1+width])
                i1 += width
            i0 += height

    else:
        raise ValueError(
            f"invalid attack type value {rand_type}, must be one of: None, 'A', or 'B'")

    return blocks


def extract_prnu_var(imgs: list, r: int, t_low: float, t_up: float, rand_type: str = None, **kwargs) -> np.ndarray:
    """
    Extract prnu noise from a list of images of same size, using one of the variance-based methods.

    :param imgs: list of np.ndarray of shape (h, w, ch) and type np.uint8 used to extract prnu
    :param r: number of block to use for noise extraction
    :param t_low: lowest mean luminance value for which a block is accepted
    :param t_up: highest mean luminance value for which a block is accepted
    :param rand_type: specify the variance-based attack type: None for <Var>, "A" for <randVar-A>, or "B" for <randVar-B>

    kwargs
    ------
    :param block_shape: (l,w) shape of blocks the images are divided into if attack type is <Var> or <randVar-A>
    :param R: number of lowest variance blocks to retain in <ranVar-*> attacks for randomized selection of r blocks later
    :param min_block_shape: (l_min, w_min) as above, used to compute randomized blocks in <randVar-B> attacks
    :param max_block_shape: (l_max, w_max) as above, used to compute randomized blocks in <randVar-B> attacks
    :param seed: to change default rng seed

    :return: prnu noise
    """

    rng = np.random.default_rng(kwargs.get("seed", 42))

    blocks = generate_blocks(imgs[0].shape, rand_type, rng, **kwargs)

    K = np.empty((imgs[0].shape[0], imgs[0].shape[1]))

    # Compute variance for each block
    for b in blocks:
        var = list()
        for im in imgs:
            im_gray = prnu.rgb2gray(im)

            mean = np.mean(im_gray[b])
            if t_low <= mean <= t_up:
                var.append([np.var(im_gray[b]), im[b]])

        var.sort(key=lambda a: a[0])
        if rand_type in ["A", "B"]:
            if "R" not in kwargs:
                raise TypeError(
                    f"missing keyward-only argument 'R' for randomized variance attack")
            var = rng.choice(var[:kwargs["R"]], r, replace=False)
        else:
            var = np.array(var[:r], copy=False, dtype=object)

        K[b] = prnu.extract_multiple_aligned(var.T[1], levels=1)
    return K
