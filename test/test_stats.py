# Copyright (c) 2021 Marco Frassineti
# Copyright (c) 2018 Nicolò Bonettini, Paolo Bestagini, Luca Bondi
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

import unittest
import os
import numpy as np
from PIL import Image

from prnu_copy_attack.extract import extract_prnu_single
from prnu_copy_attack.utils import cut_center

import prnu_copy_attack.stats as stats


def load_im(name):
    path = os.path.join(os.path.dirname(__file__), "data", name)
    return np.asanyarray(Image.open(path))


class TestStats(unittest.TestCase):

    def test_crosscorr2d(self):
        im = load_im('prnu1.jpg')[:1000, :800]

        w_all = extract_prnu_single(im)

        y_os, x_os = 300, 150
        w_cut = w_all[y_os:, x_os:]

        cc = stats.crosscorr2d(w_cut, w_all)

        max_idx = np.unravel_index(cc.argmax(), cc.shape)

        peak_y = cc.shape[0] - 1 - max_idx[0]
        peak_x = cc.shape[1] - 1 - max_idx[1]

        peak_height = cc[max_idx]

        self.assertSequenceEqual((peak_y, peak_x), (y_os, x_os))
        self.assertTrue(np.allclose(peak_height, 666995.0))

    def test_pce(self):
        im = load_im('prnu1.jpg')[:500, :400]

        w_all = extract_prnu_single(im)

        y_os, x_os = 5, 8
        w_cut = w_all[y_os:, x_os:]

        cc1 = stats.crosscorr2d(w_cut, w_all)
        cc2 = stats.crosscorr2d(w_all, w_cut)

        pce1 = stats.pce(cc1)
        pce2 = stats.pce(cc2)

        self.assertSequenceEqual(
            pce1['peak'], (im.shape[0] - y_os - 1, im.shape[1] - x_os - 1))
        self.assertTrue(np.allclose(pce1['pce'], 134611.58644973233))

        self.assertSequenceEqual(pce2['peak'], (y_os - 1, x_os - 1))
        self.assertTrue(np.allclose(pce2['pce'], 134618.03404934643))

    def test_detection(self):
        nat = load_im('nat0.jpg')
        ff1 = load_im('ff0.jpg')
        ff2 = load_im('ff1.jpg')

        nat = cut_center(nat, (500, 500, 3))
        ff1 = cut_center(ff1, (500, 500, 3))
        ff2 = cut_center(ff2, (500, 500, 3))

        w = extract_prnu_single(nat)
        k1 = extract_prnu_single(ff1)
        k2 = extract_prnu_single(ff2)

        pce1 = [{}] * 4
        pce2 = [{}] * 4

        for rot_idx in range(4):
            cc1 = stats.crosscorr2d(k1, np.rot90(w, rot_idx))
            pce1[rot_idx] = stats.pce(cc1)

            cc2 = stats.crosscorr2d(k2, np.rot90(w, rot_idx))
            pce2[rot_idx] = stats.pce(cc2)

        best_pce1 = np.max([p['pce'] for p in pce1])
        best_pce2 = np.max([p['pce'] for p in pce2])

        self.assertGreater(best_pce1, best_pce2)
