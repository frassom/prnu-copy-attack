# Copyright (c) 2021 Marco Frassineti
# Copyright (c) 2018 Nicolò Bonettini, Paolo Bestagini, Luca Bondi
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

import utils

from .noise import extract_noise
from .noise import extract_noise_compact
from .noise import wiener_dft
from .noise import zero_mean_total
from .noise import inten_sat_compact


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


def extract_prnu(imgs, levels=4, sigma=5, processes=None):
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

    RPsum = np.zeros(imgs[0].shape, np.float32)
    NN = np.zeros(imgs[0].shape, np.float32)

    # Multi process
    if processes is None or processes > 1:
        arglist = []
        for im in imgs:
            arglist.append((im, levels, sigma))

        with Pool(processes=processes) as pool:
            nni = pool.map(inten_sat_compact, arglist)
            for ni in nni:
                NN += ni
            del nni

            wi_list = pool.map(extract_noise_compact, arglist)
            for wi in wi_list:
                RPsum += wi
            del wi_list

    # Single process
    else:
        for im in imgs:
            args = (im, levels, sigma)
            RPsum += extract_noise_compact(args)
            NN += inten_sat_compact(args)

    K = RPsum / (NN + 1)
    K = utils.rgb2gray(K)
    K = zero_mean_total(K)
    K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)

    return K


def extract_prnu_var(
        imgs,
        blocks,
        r,
        lum_range=(30, 220),
        R=None,
        levels=4,
        sigma=5,
        processes=None,
        rnd=None):
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
        number of lowest variance blocks to retain in ranVar attacks for randomized selection of r blocks.
    levels : int, optional
        Number of wavelet decomposition levels.
    sigma : float, optional
        Estimated noise power.
    processes : int, optional
        Number of parallel processes.
    rng : numpy.random.Generator or int or None, optional
        see utils.get_rng .

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

    rng = utils.get_rng(rnd)

    K = np.empty(imgs[0].shape[0:2], dtype=np.float32)

    print("Choose images by minimum variance")

    grays = dict()
    imgs_used_idx = []
    for b in tqdm(blocks):
        var = []
        for im_idx, im in enumerate(imgs):
            if im_idx not in grays:
                grays[im_idx] = utils.rgb2gray(im)
            im_gray = grays[im_idx]

            mean = np.mean(im_gray[b])
            if lum_range[0] <= mean <= lum_range[1]:
                var.append([im_gray[b].var(), im_idx])

        var.sort(key=lambda a: a[0])

        # Choose the first r (or R) images by minimum variance (and randomize)
        if R is None:
            var = var[:r]
            var = np.asarray(var, dtype=object).T[1]
        else:
            var = var[:R]
            var = np.asarray(var, dtype=object).T[1]
            var = rng.choice(var, r, replace=False)

        imgs_used_idx.append(var)

    cache = dict()

    if processes is None or processes > 1:
        unique_imgs = np.unique(imgs_used_idx)

        arglist = []
        for im_idx in unique_imgs:
            arglist.append((imgs[im_idx], levels, sigma))

        with Pool(processes=processes) as pool:
            print("Compute noise")
            res = pool.imap(extract_noise_compact, arglist)
            w_list = list(tqdm(res, total=len(arglist)))

            print("Compute intensity-saturation")
            res = pool.imap(inten_sat_compact, arglist)
            nn_list = list(tqdm(res, total=len(arglist)))

        # Build cache dictionary
        for im_idx, w, nn in zip(unique_imgs, w_list, nn_list):
            cache[im_idx] = {"W": w, "NN": nn}
    else:
        for im_idx in tqdm(np.unique(imgs_used_idx)):
            args = (imgs[im_idx], levels, sigma)
            cache[im_idx] = {
                "W": extract_noise_compact(args),
                "NN": inten_sat_compact(args)
            }

    # Finally compute K for each block
    print("Extracting prnu")
    for imgs_idx, b in zip(tqdm(imgs_used_idx), blocks):
        RPsum = np.zeros(imgs[0].shape, dtype=np.float32)
        NN = np.zeros(imgs[0].shape, dtype=np.float32)
        for im_idx in imgs_idx:
            RPsum += cache[im_idx]["W"]
            NN += cache[im_idx]["NN"]

        K_full = RPsum / (NN + 1)
        K_full = utils.rgb2gray(K_full)
        K_full = zero_mean_total(K_full)
        K_full = wiener_dft(K_full, K_full.std(ddof=1)).astype(np.float32)
        K[b] = K_full[b]

    return K
