# Copyright (c) 2021 Marco Frassineti
# Copyright (c) 2018 Nicolò Bonettini, Paolo Bestagini, Luca Bondi
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

from PIL import Image
import numpy as np
import unittest
import os

import prnu_copy_attack.prnu.extract as extract


def load_im(name):
    path = os.path.join(os.path.dirname(__file__), "data", name)
    return np.array(Image.open(path))


class TestExtract(unittest.TestCase):
    def test_extract_same_shape(self):
        im = load_im('nat0.jpg')[:400, :500]
        w = extract.extract_single(im)
        self.assertSequenceEqual(w.shape, im.shape[:2])

    def test_extract_multiple_multiprocess(self):
        im1 = load_im('prnu1.jpg')[:400, :500]
        im2 = load_im('prnu2.jpg')[:400, :500]

        imgs = [im1, im2]

        k_st = extract.extract_multiple_aligned(imgs, processes=1)
        k_mt = extract.extract_multiple_aligned(imgs, processes=2)

        self.assertTrue(np.allclose(k_st, k_mt, atol=1e-6))

    def test_crosscorr2d(self):
        im = load_im('prnu1.jpg')[:1000, :800]

        w_all = extract.extract_single(im)

        y_os, x_os = 300, 150
        w_cut = w_all[y_os:, x_os:]

        cc = extract.crosscorr2d(w_cut, w_all)

        max_idx = np.unravel_index(cc.argmax(), cc.shape)

        peak_y = cc.shape[0] - 1 - max_idx[0]
        peak_x = cc.shape[1] - 1 - max_idx[1]

        peak_height = cc[max_idx]

        self.assertSequenceEqual((peak_y, peak_x), (y_os, x_os))
        self.assertTrue(np.allclose(peak_height, 666995.0))

    def test_pce(self):
        im = load_im('prnu1.jpg')[:500, :400]

        w_all = extract.extract_single(im)

        y_os, x_os = 5, 8
        w_cut = w_all[y_os:, x_os:]

        cc1 = extract.crosscorr2d(w_cut, w_all)
        cc2 = extract.crosscorr2d(w_all, w_cut)

        pce1 = extract.pce(cc1)
        pce2 = extract.pce(cc2)

        self.assertSequenceEqual(
            pce1['peak'], (im.shape[0] - y_os - 1, im.shape[1] - x_os - 1))
        self.assertTrue(np.allclose(pce1['pce'], 134611.58644973233))

        self.assertSequenceEqual(pce2['peak'], (y_os - 1, x_os - 1))
        self.assertTrue(np.allclose(pce2['pce'], 134618.03404934643))

    def test_gt(self):
        cams = ['a', 'b', 'c', 'd']
        nat = ['a', 'a', 'b', 'b', 'c', 'c', 'c']

        gt = extract.gt(cams, nat)

        self.assertTrue(np.allclose(gt, [[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, ], [0, 0, 0, 0, 1, 1, 1],
                                         [0, 0, 0, 0, 0, 0, 0, ]]))

    def test_stats(self):
        gt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        cc = np.array([[0.5, 0.2, 0.1], [0.1, 0.7, 0.1], [0.4, 0.3, 0.9]])

        stats = extract.stats(cc, gt)

        self.assertTrue(np.allclose(stats['auc'], 1))
        self.assertTrue(np.allclose(stats['eer'], 0))
        self.assertTrue(np.allclose(stats['tpr'][-1], 1))
        self.assertTrue(np.allclose(stats['fpr'][-1], 1))
        self.assertTrue(np.allclose(stats['tpr'][0], 0))
        self.assertTrue(np.allclose(stats['fpr'][0], 0))

    def test_detection(self):
        nat = load_im('nat0.jpg')
        ff1 = load_im('ff0.jpg')
        ff2 = load_im('ff1.jpg')

        nat = extract.cut_center(nat, (500, 500, 3))
        ff1 = extract.cut_center(ff1, (500, 500, 3))
        ff2 = extract.cut_center(ff2, (500, 500, 3))

        w = extract.extract_single(nat)
        k1 = extract.extract_single(ff1)
        k2 = extract.extract_single(ff2)

        pce1 = [{}] * 4
        pce2 = [{}] * 4

        for rot_idx in range(4):
            cc1 = extract.crosscorr2d(k1, np.rot90(w, rot_idx))
            pce1[rot_idx] = extract.pce(cc1)

            cc2 = extract.crosscorr2d(k2, np.rot90(w, rot_idx))
            pce2[rot_idx] = extract.pce(cc2)

        best_pce1 = np.max([p['pce'] for p in pce1])
        best_pce2 = np.max([p['pce'] for p in pce2])

        self.assertGreater(best_pce1, best_pce2)
