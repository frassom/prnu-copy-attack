"""
MIT License

Copyright (c) 2021 Marco Frassineti
Copyright (c) 2018 Nicolò Bonettini, Paolo Bestagini, Luca Bondi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
sys.path.insert(0, '..')
import os
import unittest

import numpy as np
from PIL import Image

import prnu.extract as prnu
import prnu.extract_var as prnu_var

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


class TestPrnu(unittest.TestCase):

    def test_extract(self):
        im1 = np.array(Image.open('data/nat0.jpg'))[:400, :500]
        w = prnu.extract_single(im1)
        self.assertSequenceEqual(w.shape, im1.shape[:2])

    def test_extract_multiple(self):
        im1 = np.asarray(Image.open('data/prnu1.jpg'))[:400, :500]
        im2 = np.asarray(Image.open('data/prnu2.jpg'))[:400, :500]

        imgs = [im1, im2]

        k_st = prnu.extract_multiple_aligned(imgs, processes=1)
        k_mt = prnu.extract_multiple_aligned(imgs, processes=2)

        self.assertTrue(np.allclose(k_st, k_mt, atol=1e-6))

    def test_crosscorr2d(self):
        im = np.asarray(Image.open('data/prnu1.jpg'))[:1000, :800]

        w_all = prnu.extract_single(im)

        y_os, x_os = 300, 150
        w_cut = w_all[y_os:, x_os:]

        cc = prnu.crosscorr2d(w_cut, w_all)

        max_idx = np.argmax(cc.flatten())
        max_y, max_x = np.unravel_index(max_idx, cc.shape)

        peak_y = cc.shape[0] - 1 - max_y
        peak_x = cc.shape[1] - 1 - max_x

        peak_height = cc[max_y, max_x]

        self.assertSequenceEqual((peak_y, peak_x), (y_os, x_os))
        self.assertTrue(np.allclose(peak_height, 666995.0))

    def test_pce(self):
        im = np.asarray(Image.open('data/prnu1.jpg'))[:500, :400]

        w_all = prnu.extract_single(im)

        y_os, x_os = 5, 8
        w_cut = w_all[y_os:, x_os:]

        cc1 = prnu.crosscorr2d(w_cut, w_all)
        cc2 = prnu.crosscorr2d(w_all, w_cut)

        pce1 = prnu.pce(cc1)
        pce2 = prnu.pce(cc2)

        self.assertSequenceEqual(pce1['peak'], (im.shape[0] - y_os - 1, im.shape[1] - x_os - 1))
        self.assertTrue(np.allclose(pce1['pce'], 134611.58644973233))

        self.assertSequenceEqual(pce2['peak'], (y_os - 1, x_os - 1))
        self.assertTrue(np.allclose(pce2['pce'], 134618.03404934643))

    def test_gt(self):
        cams = ['a', 'b', 'c', 'd']
        nat = ['a', 'a', 'b', 'b', 'c', 'c', 'c']

        gt = prnu.gt(cams, nat)

        self.assertTrue(np.allclose(gt, [[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, ], [0, 0, 0, 0, 1, 1, 1],
                                         [0, 0, 0, 0, 0, 0, 0, ]]))

    def test_stats(self):
        gt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        cc = np.array([[0.5, 0.2, 0.1], [0.1, 0.7, 0.1], [0.4, 0.3, 0.9]])

        stats = prnu.stats(cc, gt)

        self.assertTrue(np.allclose(stats['auc'], 1))
        self.assertTrue(np.allclose(stats['eer'], 0))
        self.assertTrue(np.allclose(stats['tpr'][-1], 1))
        self.assertTrue(np.allclose(stats['fpr'][-1], 1))
        self.assertTrue(np.allclose(stats['tpr'][0], 0))
        self.assertTrue(np.allclose(stats['fpr'][0], 0))

    def test_detection(self):
        nat = np.asarray(Image.open('data/nat0.jpg'))
        ff1 = np.asarray(Image.open('data/ff0.jpg'))
        ff2 = np.asarray(Image.open('data/ff1.jpg'))

        nat = prnu.cut_center(nat, (500, 500, 3))
        ff1 = prnu.cut_center(ff1, (500, 500, 3))
        ff2 = prnu.cut_center(ff2, (500, 500, 3))

        w = prnu.extract_single(nat)
        k1 = prnu.extract_single(ff1)
        k2 = prnu.extract_single(ff2)

        pce1 = [{}] * 4
        pce2 = [{}] * 4

        for rot_idx in range(4):
            cc1 = prnu.crosscorr2d(k1, np.rot90(w, rot_idx))
            pce1[rot_idx] = prnu.pce(cc1)

            cc2 = prnu.crosscorr2d(k2, np.rot90(w, rot_idx))
            pce2[rot_idx] = prnu.pce(cc2)

        best_pce1 = np.max([p['pce'] for p in pce1])
        best_pce2 = np.max([p['pce'] for p in pce2])

        self.assertGreater(best_pce1, best_pce2)


class TestPrnuVar(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
