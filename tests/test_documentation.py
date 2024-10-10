# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import faiss


class TestDocumentation(unittest.TestCase):

    def test_io_error(self):
        index = faiss.IndexFlatL2(32)

        self.assertTrue("Adds vectors to the index" in index.add.__doc__)
