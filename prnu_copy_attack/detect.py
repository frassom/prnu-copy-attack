# Copyright (c) 2021 Marco Frassineti
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

from multiprocessing import Pool
from itertools import chain

import numpy as np
from scipy.stats import linregress, norm as normal
import matplotlib.pyplot as plt

import extract
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


def _compute_cc(args):
    # Helper function for multiprocessing
    im, im_test, w_test, k, block_shape = args

    w = extract.extract_noise(im)
    w = utils.rgb2gray(w)
    im = utils.rgb2gray(im)

    cc_est = crosscorr_est(im, w, im_test, w_test, k, block_shape)
    cc = crosscorr(w, w_test)

    return cc, cc_est


def triangle_test(im_test, imgs_pub, imgs_priv, k, block_shape=(128, 128), processes=None):
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
    imgs_priv : list of numpy.ndarray
        Images from the private dataset of the attacked camera,
        used to estimate the reference line parameters and simulate H0.
        Must not contains images used to estimate k.
    k : numpy.ndarray
        Camera fingerprint.
        This estimate should be computed with images not in imgs, imgs_fit or imgs_est.
    block_shape : tuple of int, optional
        Shape of the block to compute the estimate of the mutual-content factor.
        Recommended shape for the blocks is between (64, 64) and (256, 256).
    proc : int, optional
        Number of processes to be used while computing the correlations.

    Returns
    -------
    TriangleTestResult
        Object containing the results.
    """

    w_test = extract.extract_noise(im_test)
    w_test = utils.rgb2gray(w_test)
    im_test = utils.rgb2gray(im_test)

    arglist = [(im, im_test, w_test, k, block_shape)
               for im in chain(imgs_pub, imgs_priv)]

    if processes is None or processes > 1:
        with Pool(processes=processes) as pool:
            corr = pool.map(_compute_cc, arglist)

    else:
        corr = [_compute_cc(args) for args in arglist]

    corr_pub = np.asarray(corr[:len(imgs_pub)], dtype=float)
    corr_priv = np.asarray(corr[len(imgs_pub):], dtype=float)

    return TriangleTestResult(corr_pub, corr_priv)


class TriangleTestResult:
    """
    Class containing results from the Triangle Test.

    Parameters
    ----------
    corr_pub_all : numpy.ndarray
        Correlations and estimated correlations from the public dataset.
    corr_priv_all : numpy.ndarray
        Correlations and estimated correlations from the private dataset.
    """

    def __init__(self, corr_pub_all, corr_priv_all):
        self.corr_pub = corr_pub_all[:, 0]
        self.corr_pub_est = corr_pub_all[:, 1]
        self.corr_priv = corr_priv_all[:, 0]
        self.corr_priv_est = corr_priv_all[:, 1]

        # Number of values used to estimate the reference line
        # the remaining are used for the normal fit
        line_chunk = self.corr_priv.size * .75
        line_chunk = int(line_chunk)

        corr_line = self.corr_priv[:line_chunk]
        corr_fit = self.corr_priv[line_chunk:]
        corr_est_line = self.corr_priv_est[:line_chunk]
        corr_est_fit = self.corr_priv_est[line_chunk:]

        # Compute reference line
        lr = linregress(corr_est_line, corr_line)
        self.slope = lr.slope
        self.intercept = lr.intercept

        # Compute normal fit for the statistic
        d_priv = corr_fit - self.slope * corr_est_fit - self.intercept
        self.loc, self.scale = normal.fit(d_priv)

        # Test statistic
        self.stat = self.corr_pub - self.slope * self.corr_pub_est - self.intercept

    def threshold(self, p_fa=1e-4):
        """
        Computes the threshold constraining the false alarm probability p_fa.
        """

        return normal.isf(p_fa, self.loc, self.scale)

    def plot_stat(self, mode='h', idx_used=None, bins=100, ax=None):
        """
        Plot the test statistic with a scatter plot plotting also the threshold,
        or in an instogram plotting the estimated pdf.

        Parameters
        ----------
        mode : str, optional
            Type of graph to show 's' for scatter, 'h' for histogram.
        idx_used : list of int, optional
            List containings the idices of the images used to compute the
            fingerprint estimate by the attacker (not known in a real scenario).
        bins : int, optional
            Bins for histogram plot.
        ax : Axes, optional
        """

        if ax is None:
            plt.figure()
            ax = plt.axes()

        if mode == 'h':  # histogram plot
            x = np.linspace(self.stat.min(), self.stat.max(), 100)
            ax.plot(x, normal.pdf(x, self.loc, self.scale),
                    color='gray', linewidth=.7)

            if idx_used is None:
                ax.hist(self.stat, bins=bins)
            else:
                idx_notused = np.delete(np.arange(self.stat.size), idx_used)
                ax.hist([self.stat[idx_used], self.stat[idx_notused]],
                        bins=bins,
                        label=["Images used by Eve", "Images not used by Eve"],
                        stacked=True)
                ax.legend()

            ax.set_xlabel("test stat")

        elif mode == 's':  # scatter plot
            x = np.arange(self.stat.size)
            ax.plot(x, [self.threshold()]*x.size,
                    color='gray', linewidth=.7)

            if idx_used is None:
                ax.scatter(x, self.stat, s=1)
            else:
                ax.scatter(x[idx_used], self.stat[idx_used],
                           s=1, label="Images used by Eve")
                idx_notused = np.delete(np.arange(self.stat.size), idx_used)
                ax.scatter(x[idx_notused], self.stat[idx_notused],
                           s=1, label="Images not used by Eve")
                ax.legend()

            ax.set_xlabel("Images")
            ax.set_ylabel("test stat")

    def plot_corr(self, idx_used=None, ax=None):
        """
        Plot true corr against estimated corr and the reference line.

        Parameters
        ----------
        idx_used : list of int
            List containings the idices of the images used to compute the
            fingerprint estimate by the attacker (not known in a real scenario).
        ax : Axes, optional
        """

        if ax is None:
            plt.figure()
            ax = plt.axes()

        if idx_used is None:
            ax.scatter(self.corr_pub_est, self.corr_pub,
                       s=1, marker=',', label="Public images")
            ax.scatter(self.corr_priv_est, self.corr_priv,
                       s=1, marker=',', label="Private images")
        else:
            ax.scatter(self.corr_pub_est[idx_used], self.corr_pub[idx_used],
                       s=1, marker=',', label="Images used by Eve")
            idx_notused = np.delete(
                np.arange(self.corr_pub.size), idx_used)
            ax.scatter(self.corr_pub_est[idx_notused], self.corr_pub[idx_notused],
                       s=1, marker=',', label="Images not used by Eve")

        x = np.array(ax.get_xlim())
        y = self.intercept + self.slope * x
        ax.plot(x, y, color="gray", linewidth=.7)

        ax.set_xlabel("estimated crosscorr")
        ax.set_ylabel("crosscorr")
        ax.legend()
