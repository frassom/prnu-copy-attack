# Copyright (c) 2021 Marco Frassineti
# Copyright (c) 2018 Nicolò Bonettini, Paolo Bestagini, Luca Bondi
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import filters
import pywt


def extract_noise(im, levels=4, sigma=5):
    """
    NoiseExtract as from Binghamton toolbox.

    Parameters
    ----------
    im : numpy.ndarray
        Grayscale or color image of type np.uint8.
    levels : int, optional
        Number of wavelet decomposition levels.
    sigma : float, optional
        Estimated noise power.

    Retunrs
    -------
    numpy.ndarray
        Noise residual.
    """

    if not im.dtype == np.uint8:
        raise ValueError("im.dtype must be np.uint8")

    if not im.ndim in [2, 3]:
        raise ValueError("im.ndim must be 2 or 3")

    im = im.astype(np.float32)

    noise_var = sigma ** 2

    if im.ndim == 2:
        im.shape += (1,)

    W = np.zeros(im.shape, np.float32)

    for ch in range(im.shape[2]):

        wlet = None
        while wlet is None and levels > 0:
            try:
                wlet = pywt.wavedec2(im[:, :, ch], 'db4', level=levels)
            except ValueError:
                levels -= 1
                wlet = None
        if wlet is None:
            raise ValueError(
                f'Impossible to compute Wavelet filtering for input size: {im.shape}')

        wlet_details = wlet[1:]

        wlet_details_filter = [None] * len(wlet_details)
        # Cycle over Wavelet levels 1:levels-1
        for wlet_level_idx, wlet_level in enumerate(wlet_details):
            # Cycle over H,V,D components
            level_coeff_filt = [None] * 3
            for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                level_coeff_filt[wlet_coeff_idx] = \
                    wiener_adaptive(wlet_coeff, noise_var)
            wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

        # Set filtered detail coefficients for Levels > 0 ---
        wlet[1:] = wlet_details_filter

        # Set to 0 all Level 0 approximation coefficients ---
        wlet[0][...] = 0

        # Invert wavelet transform ---
        wrec = pywt.waverec2(wlet, 'db4')
        try:
            W[:, :, ch] = wrec
        except ValueError:
            W = np.zeros(wrec.shape[:2] + (im.shape[2],), np.float32)
            W[:, :, ch] = wrec

    if W.shape[2] == 1:
        W.shape = W.shape[:2]

    W = W[:im.shape[0], :im.shape[1]]

    return W


def extract_noise_compact(args):
    """
    Extract residual, multiplied by the image.
    Useful to save memory in multiprocessing operations.

    Parameters
    ----------
    args : (im, levels, sigma)
        See noise_extract for usage.

    Returns
    -------
    numpy.ndarray
        Noise residual, multiplied by the image.
    """

    w = extract_noise(*args)
    im = args[0]
    return (w * im / 255.).astype(np.float32)


def wiener_dft(im, sigma):
    """
    Adaptive Wiener filter applied to the 2D FFT of the image.

    Parameters
    ----------
    im : numpy.ndarray
        Multidimensional array.
    sigma : float
        Estimated noise power.

    Returns
    -------
    numpy.ndarray
        Filtered version of input im.
    """

    noise_var = sigma ** 2
    h, w = im.shape

    im_noise_fft = fft2(im)
    im_noise_fft_mag = np.abs(im_noise_fft / (h * w) ** .5)

    im_noise_fft_mag_noise = wiener_adaptive(im_noise_fft_mag, noise_var)

    zeros_y, zeros_x = np.nonzero(im_noise_fft_mag == 0)

    im_noise_fft_mag[zeros_y, zeros_x] = 1
    im_noise_fft_mag_noise[zeros_y, zeros_x] = 0

    im_noise_fft_filt = im_noise_fft * im_noise_fft_mag_noise / im_noise_fft_mag
    im_noise_filt = np.real(ifft2(im_noise_fft_filt))

    return im_noise_filt.astype(np.float32)


def zero_mean(im):
    """
    ZeroMean called with the 'both' argument, as from Binghamton toolbox.

    Parameters
    ----------
    im : numpy.ndarray
        Multidimensional array.

    Returns
    -------
    numpy.ndarray
        Zero mean version of input im.
    """

    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    # Subtract the 2D mean from each color channel ---
    ch_mean = im.mean(axis=0).mean(axis=0)
    ch_mean.shape = (1, 1, ch)
    i_zm = im - ch_mean

    # Compute the 1D mean along each row and each column, then subtract ---
    row_mean = i_zm.mean(axis=1)
    col_mean = i_zm.mean(axis=0)

    row_mean.shape = (h, 1, ch)
    col_mean.shape = (1, w, ch)

    i_zm_r = i_zm - row_mean
    i_zm_rc = i_zm_r - col_mean

    # Restore the shape ---
    if im.shape[2] == 1:
        i_zm_rc.shape = im.shape[:2]

    return i_zm_rc


def zero_mean_total(im):
    """
    ZeroMeanTotal as from Binghamton toolbox.

    Parameters
    ----------
    im : numpy.ndarray
        Multidimensional array.

    Returns
    -------
    numpy.ndarray
        Zero mean version of input im.
    """

    im[0::2, 0::2] = zero_mean(im[0::2, 0::2])
    im[1::2, 0::2] = zero_mean(im[1::2, 0::2])
    im[0::2, 1::2] = zero_mean(im[0::2, 1::2])
    im[1::2, 1::2] = zero_mean(im[1::2, 1::2])

    return im


def threshold(wlet_coeff_energy_avg, noise_var):
    """
    Noise variance theshold as from Binghamton toolbox.

    Parameters
    ----------
    wlet_coeff_energy_avg : numpy.ndarray
    noise_var : float

    Returns
    -------
    numpy.ndarray
        Noise variance threshold
    """

    res = wlet_coeff_energy_avg - noise_var
    return (res + np.abs(res)) / 2


def wiener_adaptive(x, noise_var, window_sizes=None):
    """
    WaveNoise as from Binghamton toolbox.
    Wiener adaptive flter aimed at extracting the noise component
    For each input pixel the average variance over a neighborhoods of different
    window sizes is first computed. The smaller average variance is taken into
    account when filtering according to Wiener.

    Parameters
    ----------
    x: numpy.ndarray
        2D matrix.
    noise_var : float
        Power spectral density of the noise we wish to extract (S).
    window_sizes : list, optional
        List of window sizes.

    Returns
    -------
    numpy.ndarray
        Wiener filtered version of input x.
    """

    if window_sizes is None:
        window_sizes = [3, 5, 7, 9]

    energy = x ** 2

    avg_win_energy = np.zeros(x.shape + (len(window_sizes),))
    for window_idx, window_size in enumerate(window_sizes):
        avg_win_energy[:, :, window_idx] = \
            filters.uniform_filter(energy, window_size, mode='constant')

    coef_var = threshold(avg_win_energy, noise_var)
    coef_var_min = np.min(coef_var, axis=2)

    x = x * noise_var / (coef_var_min + noise_var)

    return x


def inten_scale(im):
    """
    IntenScale as from Binghamton toolbox.

    Parameters
    ----------
    im : numpy.ndarray
        Image of dtype np.uint8.

    Returns
    -------
    numpy.ndarray
        Intensity scaled version of input x.
    """

    if not im.dtype == np.uint8:
        raise ValueError("im.dtype must be np.uint8")

    T = 252
    v = 6
    out = np.exp(-1 * (im - T) ** 2 / v)
    out[im < T] = im[im < T] / T

    return out


def saturation(im):
    """
    Saturation as from Binghamton toolbox.

    Parameters
    ----------
    im : numpy.ndarray
        Image of dtype np.uint8.

    Returns
    -------
    numpy.ndarray
        Saturation map from input im.
    """

    if not im.dtype == np.uint8:
        raise ValueError("im.dtype must be np.uint8")

    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    if im.max() < 250:
        return np.ones((h, w, ch))

    im_h = im - np.roll(im, (0, 1), (0, 1))
    im_v = im - np.roll(im, (1, 0), (0, 1))
    satur_map = \
        np.bitwise_not(
            np.bitwise_and(
                np.bitwise_and(
                    np.bitwise_and(
                        im_h != 0, im_v != 0
                    ), np.roll(im_h, (0, -1), (0, 1)) != 0
                ), np.roll(im_v, (-1, 0), (0, 1)) != 0
            )
        )

    max_ch = im.max(axis=0).max(axis=0)

    for ch_idx, max_c in enumerate(max_ch):
        if max_c > 250:
            satur_map[:, :, ch_idx] = \
                np.bitwise_not(
                    np.bitwise_and(
                        im[:, :, ch_idx] == max_c, satur_map[:, :, ch_idx]
                    )
            )

    return satur_map


def inten_sat_compact(args):
    """
    Memory saving version of inten_scale followed by saturation.
    Useful for multiprocessing.

    Parameters
    ----------
    im : numpy.ndarray
        Image of dtype np.uint8.

    Returns
    -------
    numpy.ndarray
        Intensity scale and saturation of input.
    """
    return ((inten_scale(args[0]) * saturation(args[0])) ** 2).astype(np.float32)
