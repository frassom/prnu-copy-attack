# Copyright (c) 2021 Marco Frassineti
# Copyright (c) 2018 Nicolò Bonettini, Paolo Bestagini, Luca Bondi
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

import numpy as np
import scipy as sp


def crosscorr(x, y):
    """
    Compute normalized cross-correlation between two images.

    Parameters
    ----------
    x : numpy.ndarray
        2D matrix of size (h,w).
    y : numpy.ndarray
        2D matrix of size (h,w).

    Returns
    -------
    float
    """

    if x.shape != y.shape:
        raise ValueError("input ndarray must have the same shape")

    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return sp.signal.correlate2d(x, y, mode="valid")[0, 0] / norm


def crosscorr2d(x, y) -> np.ndarray:
    """
    Compute 2D cross-correlation matrix.

    Parameters
    ----------
    x : numpy.ndarray
        2D matrix of size (h1,w1).
    y : numpy.ndarray
        2D matrix of size (h2,w2).

    Returns
    -------
    numpy.ndarray
        2D matrix of size (max(h1,h2),max(w1,w2)).
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("input ndarray must be two dimensional")

    max_height = max(x.shape[0], y.shape[0])
    max_width = max(x.shape[1], y.shape[1])

    x -= x.mean()
    y -= y.mean()

    x = np.pad(x, [(0, max_height - x.shape[0]), (0, max_width - x.shape[1])],
               mode='constant',
               constant_values=0)
    y = np.pad(y, [(0, max_height - y.shape[0]), (0, max_width - y.shape[1])],
               mode='constant',
               constant_values=0)

    k1_fft = np.fft.fft2(x)
    k2_fft = np.fft.fft2(np.rot90(y, 2))

    return np.real(np.fft.ifft2(k1_fft * k2_fft)).astype(np.float32)


def pce(cc, radius=2) -> dict:
    """
    Computes PCE position and value.

    Parameters
    ----------
    cc : numpy.ndarray
        Cross-correlation matrix as from crosscorr2d.
    radius : int, optional
        Radius around the peak to be ignored while computing floor energy.

    Returns
    -------
    dict
        'peak': Peak coordinates (x, y).
        'pce': Peak to correlation energy.
        'cc': Cross-correlation value at peak position.
    """

    assert (cc.ndim == 2)
    assert (isinstance(radius, int))

    max_idx = np.unravel_index(cc.argmax(), cc.shape)

    peak_height = cc[max_idx]

    cc_nopeaks = cc.copy()
    cc_nopeaks[max_idx[0] - radius:max_idx[0] + radius,
               max_idx[1] - radius:max_idx[1] + radius] = 0

    pce_energy = np.mean(cc_nopeaks ** 2)

    return {
        'peak': max_idx,
        'pce': (peak_height ** 2) / pce_energy * np.sign(peak_height),
        'cc': peak_height
    }


def psnr(x, y):
    """
    Compute the Peak Signal to Noise Ratio between the two images.
    If im1 == im2 np.inf is returned.

    Parameters
    ----------
    x, y : numpy.ndarray
        Multidimensional array.

    Returns
    -------
    psnr : float
    """

    mse = np.mean(np.square(x.astype(float) - y))
    if mse == 0:
        return np.inf
    else:
        return 10 * np.log10(255**2 / mse)
