# Copyright (c) 2021 Marco Frassineti
# Copyright (c) 2018 Nicolò Bonettini, Paolo Bestagini, Luca Bondi
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

import utils

from .noise import extract_noise
from .noise import extract_noise_compact
from .noise import wiener_dft
from .noise import zero_mean_total
from .noise import inten_sat_compact
from .noise import inten_scale
from .noise import saturation


def extract_prnu_single(im, levels=4, sigma=5, wdft_sigma=0):
    """
    Extract PRNU noise from a single image.

    Parameters
    ----------
    im : numpy.ndarray
        Grayscale or color image, np.uint8.
    levels : int, optional
        Number of wavelet decomposition levels.
    sigma : float, optional
        Estimated noise power.
    wdft_sigma : float, optional
        Estimated DFT noise power.

    Returns
    -------
    K : numpy.ndarray
        PRNU noise.
    """

    W = extract_noise(im, levels, sigma)
    W = utils.rgb2gray(W)
    W = zero_mean_total(W)
    W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
    W = wiener_dft(W, W_std).astype(np.float32)

    return W


def extract_prnu(imgs, levels=4, sigma=5, processes=None, batch_size=cpu_count()):
    """
    Extract PRNU from a list of images. Images are supposed to be the same
    size and properly oriented.

    Parameters
    ----------
    imgs : list
        List of images of size (H,W,Ch) and type np.uint8.
    levels : int, optional
        Number of wavelet decomposition levels.
    sigma : float, optional
        Estimated noise power.
    processes : int, optional
        Number of parallel processes.
    batch_size : int, optional
        Number of parallel processed images.

    Returns
    -------
    K : numpy.ndarray
        PRNU noise.
    """

    if not isinstance(imgs[0], np.ndarray):
        raise ValueError("imgs must cointains np.ndarray")

    if not imgs[0].ndim == 3:
        raise ValueError("images must have 3 channels")

    if not imgs[0].dtype == np.uint8:
        raise ValueError("images type must be np.uint8")

    h, w, ch = imgs[0].shape

    RPsum = np.zeros((h, w, ch), np.float32)
    NN = np.zeros((h, w, ch), np.float32)

    # Multi process
    if processes is None or processes > 1:
        args_list = []
        for im in imgs:
            args_list += [(im, levels, sigma)]

        pool = Pool(processes=processes)

        for i in np.arange(0, len(imgs), batch_size):
            nni = pool.map(inten_sat_compact, args_list[i:i + batch_size])
            for ni in nni:
                NN += ni
            del nni

        for i in np.arange(0, len(imgs), batch_size):
            wi_list = pool.map(extract_noise_compact,
                               args_list[i:i + batch_size])
            for wi in wi_list:
                RPsum += wi
            del wi_list

        pool.close()

    # Single process
    else:
        for im in imgs:
            RPsum += extract_noise_compact((im, levels, sigma))
            NN += (inten_scale(im) * saturation(im)) ** 2

    K = RPsum / (NN + 1)
    K = utils.rgb2gray(K)
    K = zero_mean_total(K)
    K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)

    return K


def extract_prnu_var(imgs, blocks, r, lum_range=(30, 220), R=None, rnd=None, **kwargs):
    """
    Extract prnu noise from a list of images of same size, using one of the variance-based methods

    Parameters
    ----------
    imgs : list of numpy.ndarray
        images of shape (h, w, ch) and type np.uint8 used to extract prnu
    blocks : list of (slice, slice)
        division in block of images for prnu extraction, see gen_block[_rnd]
    r : int
        number of block to use for noise extraction
    lum_range : tuple of int, optional
        (t_low, t_up) range of luminance value for which a block is accepted
    R : int, optional
        number of lowest variance blocks to retain in ranVar attacks for randomized selection of r blocks
    rng : numpy.random.Generator, optional
        the numpy random generator to use, if None the default is used
    **kwargs : dict, optional
        additional parameters to pass to extract_multiple_aligned in extract.py

    Returns
    -------
    K : numpy.ndarray
        PRNU noise.
    """

    rng = utils.get_rng(rnd)

    K = np.empty(imgs[0].shape[0:2])

    # Compute variance for each block
    for b in tqdm(blocks):
        var = list()
        for im in imgs:
            im_gray = utils.rgb2gray(im)

            mean = np.mean(im_gray[b])
            if lum_range[0] <= mean <= lum_range[1]:
                var.append([np.var(im_gray[b]), im])

        var.sort(key=lambda a: a[0])

        if R is None:
            var = np.asanyarray(var[:r], dtype=object)
        else:
            var = np.asanyarray(var[:R], dtype=object)
            var = rng.choice(var, r, replace=False)

        K[b] = extract_prnu(var.T[1], **kwargs)[b]

    return K
