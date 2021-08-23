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

import prnu_copy_attack.extract as extract


def load_im(name):
    path = os.path.join(os.path.dirname(__file__), "data", name)
    return np.asanyarray(Image.open(path))


class TestExtract(unittest.TestCase):

    def test_extract_same_shape(self):
        im = load_im('nat0.jpg')[:400, :500]
        w = extract.extract_prnu_single(im)
        self.assertSequenceEqual(w.shape, im.shape[:2])

    def test_extract_multiple_multiprocess(self):
        im1 = load_im('prnu1.jpg')[:400, :500]
        im2 = load_im('prnu2.jpg')[:400, :500]

        imgs = [im1, im2]

        k_st = extract.extract_prnu(imgs, processes=1)
        k_mt = extract.extract_prnu(imgs, processes=2)

        self.assertTrue(np.allclose(k_st, k_mt, atol=1e-6))
