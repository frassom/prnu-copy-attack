# Copyright (c) 2021 Marco Frassineti
#
# Licensed under the MIT license: https://opensource.org/licenses/MIT
# Permission is granted to use, copy, modify, and redistribute the work.
# Full license information available in the project LICENSE file.

import os
import sys
import unittest


if __name__ == '__main__':
    src_path = os.path.dirname(os.path.realpath(__file__))
    src_path = os.path.join(src_path, "..", "prnu_copy_attack")
    src_path = os.path.realpath(src_path)
    sys.path.append(src_path)

    loader = unittest.TestLoader()
    suite = loader.discover(".")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
