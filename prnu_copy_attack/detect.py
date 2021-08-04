# Copyright (c) 2021 Marco Frassineti
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

import numpy as np
from scipy.stats import linregress, norm as normal
import matplotlib.pyplot as plt
from tqdm import tqdm

import prnu
import utils
from stats import crosscorr


def att_factor(im, w, k, q=1):
    """
    Compute an estimate for the attenuation factor, in a small block.

    Parameters
    ----------
    im : numpy.ndarray
        Image from which the attenuation factor has to be estimated.
    w : numpy.ndarray
        Noise residual of im.
    k : numpy.ndarray
        Camera fingerprint.
    q : float, optional
        Fingerprint quality.

    Returns
    -------
    a : float
        Estimate for the attenuation factor.
    """

    corr = crosscorr(w, im * k)
    w_norm = np.linalg.norm(w)
    k_norm = np.linalg.norm(k)
    im_norm = np.sqrt(np.mean(im**2))
    return (w_norm * corr * q**-2) / (im_norm * k_norm)


def crosscorr_est(im, w, im_test, w_test, k, block_shape=(128, 128), q=1):
    """
    Compute the estimate of the cross-correlation between
    im and im_test using the provided camera fingerprint k.

    Parameters
    ----------
    im : numpy.ndarray
        Grayscale image known to come from camera.
    w : numpy.ndarray
        Noise residual of im.
    im_test : numpy.ndarray
        Grayscale image under test.
    w_test : numpy.ndarray
        Noise residual of im_test.
    k : numpy.ndarray
        Camera fingerprint.
    block_shape : tuple of int, optional
        As in triangle_test.
    q : float, optional
        Fingerprint quality factor in range [0, 1].

    Returns
    -------
    float
        The estimated cross-correlation.
    """

    if im.shape != k.shape or im.shape != im_test.shape:
        raise ValueError("im, k, and im_test must have matching shapes")

    # Compute the mutual-content factor (μ)
    blocks = utils.gen_blocks(im.shape, block_shape)

    num = 0
    den, den_test = 0, 0
    for b in blocks:
        a = att_factor(im[b], w[b], k[b], q)
        a_test = att_factor(im_test[b], w_test[b], k[b], q)

        num += a * a_test * np.mean(im[b] * im_test[b])
        den += a * im[b].mean()
        den_test += a_test * im_test[b].mean()

    mcf = num / (den * den_test) * len(blocks)

    return crosscorr(w, k) * crosscorr(w_test, k) * mcf * q**-2


def _compute_corr(im_test, w_test, imgs, k, block_shape):
    """
    Compute correlation and correlation estimate between images in imgs and im_test.

    Returns
    -------
    cc_est : list of float
        Estimated cross-correlation.
    cc : list of float
        Cross-correlation.
    """

    cc_est = []
    cc = []
    for im in tqdm(imgs):
        w = prnu.extract_noise(im)
        w = utils.rgb2gray(w)
        im = utils.rgb2gray(im)

        cc_est.append(crosscorr_est(im, w, im_test, w_test, k, block_shape))
        cc.append(crosscorr(w, w_test))

    return cc_est, cc


def triangle_test(im_test, imgs_pub, imgs_est, imgs_fit, k, block_shape=(128, 128)):
    """
    Compute the Triangle Test as described in Fridrich et al. [2],
    return a class containing the results.

    Parameters
    ----------
    im_test : numpy.ndarray
        Image to be tested for forgery, of type numpy.uint8.
    imgs_pub : list of numpy.ndarray
        Images from the public dataset of the attacked camera, which may
        contains images used to estimate the PRNU noise by the attacker.
    imgs_est : list of numpy.ndarray
        Images from the private dataset of the attacked camera,
        used to estimate the reference line parameters.
        Must contains images used to estimate k.
    imgs_fit : list of numpy.ndarray
        Images from the private dataset of the attacked camera,
        used to simulate H0 (estimate pdf of the test statistic).
        Must contains images used to estimate k.
    k : numpy.ndarray
        Camera fingerprint.
        This estimate should be computed with images not in imgs, imgs_fit or imgs_est.
    block_shape : tuple of int, optional
        Shape of the block to compute the estimate of the mutual-content factor.
        Recommended shape for the blocks is between (64, 64) and (256, 256).

    Returns
    -------
    TriangleTestResults
        Class containing all the results obtained by the TriangleTest.
    """

    w_test = prnu.extract_noise(im_test)
    w_test = utils.rgb2gray(w_test)
    im_test = utils.rgb2gray(im_test)

    # Correlation of private images for reference line
    cc_est_priv, cc_priv = \
        _compute_corr(im_test, w_test, imgs_est, k, block_shape)
    linreg = linregress(cc_est_priv, cc_priv)

    # Correlation of private images for fit
    cc_est, cc = _compute_corr(im_test, w_test, imgs_fit, k, block_shape)
    cc_est_priv += cc_est
    cc_priv += cc

    # Compute normal fit
    cc_est = np.asanyarray(cc_est)
    cc = np.asanyarray(cc)
    d = cc - linreg.slope * cc_est - linreg.intercept
    fit = normal.fit(d)

    # Correlation of public images
    corr_pub = _compute_corr(im_test, w_test, imgs_pub, k, block_shape)

    return TriangleTestResults(corr_pub, (cc_est_priv, cc_priv), linreg, fit)


class TriangleTestResults:
    """
    Class containing results from the Triangle Test.

    Parameters
    ----------
    corr : list of float
        True crosscorr between the tested image and images from the public dataset.
    corr_est : list of float
        Estimated crosscorr between the tested image and images from the public dataset.
    linreg : LinregressResult
        Result from scipy.stats.linregress .
    fit : tuple of float
        (loc, scale) result from scipy.stats.norm.fit .

    Attributes
    ----------
    likelihood : float
        The scaled log-likelihood computed with corr_pub.
    """

    def __init__(self, corr_pub, corr_priv, linreg, fit):
        self.corr_pub = np.asanyarray(corr_pub)
        self.corr_priv = np.asanyarray(corr_priv)
        self.slope = linreg.slope
        self.intercept = linreg.intercept
        self.loc = fit[0]
        self.scale = fit[1]

        # Compute scaled log-likelihood
        size = self.corr_priv.shape[1]
        d = self.corr_pub[1] - self.slope * self.corr_pub[0] - self.intercept
        self.likelihood = \
            np.sum(normal.logpdf(d, self.loc, self.scale)) / np.sqrt(size)

    def threshold(self, p_fa=1e-4):
        """
        Computes the threshold for the statistic,
        constraining the false alarm probability p_fa.
        """

        return normal.isf(p_fa, self.loc, self.scale)

    def is_forged(self, p_fa=1e-4):
        """Check if the image was forged (Lk < Th)."""

        return self.likelihood < self.threshold(p_fa)

    def plot(self, show_private=True, idx_used=None):
        """
        Plot true corr against estimated corr and the reference line.

        Parameters
        ----------
        show_private : bool
            Wether to plot the correlations from the private datased used
            to build the reference line.
        idx_used : list of int
            List containings the idices of the images used to compute the
            fingerprint estimate by the attacker (not known in a real scenario).
        """

        if idx_used is None:
            plt.scatter(self.corr_pub[0], self.corr_pub[1], s=1,
                        marker=',', label="Public images")
        else:
            plt.scatter(self.corr_pub[0][idx_used], self.corr_pub[1][idx_used],
                        s=1, marker=',', label="Public images used by Eve")
            idx_notused = np.delete(np.arange(self.corr_pub[0].size), idx_used)
            plt.scatter(self.corr_pub[0][idx_notused], self.corr_pub[1][idx_notused],
                        s=1, marker=',', label="Public images not used by Eve")
            plt.legend()

        if show_private:
            plt.scatter(self.corr_priv[0], self.corr_priv[1],
                        s=1, marker=',', label="Private images")
            plt.legend()

        x = np.array(plt.xlim())
        y = self.intercept + self.slope * x
        plt.plot(x, y, color="#555", linestyle='-.')

        plt.xlabel("estimated crosscorr")
        plt.ylabel("crosscorr")
        plt.show()
