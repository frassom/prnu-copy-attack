# Copyright (c) 2021 Marco Frassineti
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

import numpy as np
import scipy as sp

import stats


def copy_attack(im, k, alpha=None):
    """
    Perform a fingerprint-copy attack on im with fingerprint k

    Parameters
    ----------
    im : numpy.ndarray
        Image to forge with the fingerprint, of shape (h, w, ch).
    k : numpy.ndarray
        Camera Fingerprint to apply on im, of shape (h, w).
    alpha : float, optional
        Fingerprint strength, if None the value is estimated through compute_alpha

    Returns
    -------
    numpy.ndarray
        Forged image numpy.uint8.
    """

    if im.shape[:2] != k.shape:
        raise ValueError("im and k shapes don't match")

    if alpha is None:
        alpha = compute_alpha(im, k)

    im = im.astype(np.float32)
    im_forge = im + alpha * im * k[:, :, None]
    return im_forge.round().clip(0, 255).astype(np.uint8)


def compute_alpha(im, k, A=50):
    def psnr(alpha):
        return stats.psnr(im, copy_attack(im, k, alpha)) - A

    a = 0
    b = 0.2
    while psnr(b) >= 0:
        b += 0.2

    x0, r = sp.optimize.bisect(psnr, a, b, maxiter=100, full_output=True)

    if not r.converged:
        raise RuntimeError("Unable to find alpha, did not converge")

    return x0
