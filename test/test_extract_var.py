# Copyright (c) 2021 Marco Frassineti
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

import prnu_copy_attack.prnu.extract_var as extract_var
from PIL import Image
import numpy as np
import unittest


class TestExtractVar(unittest.TestCase):
    def test_gen_block_shape(self):
        im = np.empty((3120, 4160))
        block_shape = (16, 16)

        blocks = extract_var.gen_blocks(im.shape, block_shape)

        self.assertEqual(len(blocks),
                         im.shape[0] / block_shape[0] * im.shape[1] / block_shape[1])
        for b in blocks:
            self.assertEqual(im[b].shape, block_shape)

    def test_gen_block_no_ovrlap(self):
        im = np.zeros((512, 512), dtype=bool)

        blocks = extract_var.gen_blocks(im.shape, (16, 16))

        for b in blocks:
            self.assertFalse(im[b].all())
            im[b] = True

    def test_gen_block_rnd_shape(self):
        im = np.empty((3120, 4160))
        min_block_shape = (16, 16)
        max_block_shape = (40, 40)
        rng = np.random.default_rng()

        blocks = extract_var.gen_blocks_rnd(
            im.shape, min_block_shape, max_block_shape, rng=rng)

        for b in blocks:
            self.assertGreaterEqual(im[b].shape[0], min_block_shape[0])
            self.assertLessEqual(im[b].shape[0], max_block_shape[0])
            self.assertGreaterEqual(im[b].shape[1], min_block_shape[1])
            self.assertLessEqual(im[b].shape[1], max_block_shape[1])

    def test_gen_block_rnd_no_overlap(self):
        im = np.zeros((512, 512), dtype=bool)
        rng = np.random.default_rng()

        blocks = extract_var.gen_blocks_rnd(im.shape, rng=rng)

        for b in blocks:
            self.assertFalse(im[b].all())
            im[b] = True
