# Copyright (c) 2021 Marco Frassineti
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

import numpy as np
import scipy.optimize as opt

import stats


def copy_attack(im, k, alpha=None, blocks=None, A=50):
    """
    Perform a fingerprint-copy attack on im with fingerprint k.

    Parameters
    ----------
    im : numpy.ndarray
        Image to forge with the fingerprint, of shape (h, w, ch).
    k : numpy.ndarray
        Camera Fingerprint to apply on im, of shape (h, w).
    alpha : float or list of float, optional
        Fingerprint strength, if None the value is estimated through compute_alpha.
    blocks : list of (slice, slice), optional
        Used to compute alpha when alpha is None. If None alpha will be computed for the whole image

    Returns
    -------
    numpy.ndarray
        Forged image numpy.uint8.
    """

    if im.shape[:2] != k.shape:
        raise ValueError("im and k shapes don't match")

    if alpha is None:
        alpha = compute_alpha(im, k, blocks, A)
    elif isinstance(alpha, float):
        alpha = [alpha]

    if not isinstance(alpha, list):
        raise ValueError("invalid value of alpha")

    im = im.astype(np.float32)
    if blocks is None:
        im_forge = im + alpha[0] * im * k[:, :, None]
    else:
        im_forge = np.empty(im.shape, np.float32)
        for b, a in zip(blocks, alpha):
            im_forge[b] = im[b] + a * im[b] * k[b][:, :, None]

    return im_forge.round().clip(0, 255).astype(np.uint8)


def compute_alpha(im, k, blocks=None, A=50):
    """
    Return the value of fingerprint strength that if used on an image
    J to generate J' with copy_attack result in PSNR(J,J') == A.

    As described in:
    H. Li, W. Luo, Q. Rao, J. Huang 
        "Anti-Forensics of Camera Identification and the
         Triangle Test by Improved Fingerprint-Copy Attack."

    Parameters
    ----------
    im : numpy.ndarray
        Image to be forged with k.
    k : numpy.ndarray
        Camera fingerprint.
    blocks : list of (slice, slice), optional
    A : float, optional
        Target value of PSNR, should be in the range [47.6, 58.7].

    Returns
    -------
    float
        Fingerprint strenght(s).
    """

    if blocks is None:
        blocks = [np.s_[0:im.shape[0], 0:im.shape[1]]]

    alphas = []

    for i, b in enumerate(blocks):
        def psnr(alpha):
            return stats.psnr(im[b], copy_attack(im[b], k[b], alpha)) - A

        a1 = 0
        a2 = 0.2
        while psnr(a2) >= 0:
            a2 += 0.2

        a, r = opt.bisect(psnr, a1, a2, maxiter=100, full_output=True)
        alphas.append(a)

        if not r.converged:
            raise RuntimeError("Unable to find alpha, did not converge")

    return alphas
