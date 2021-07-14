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

    def test_doxygen_comments(self):
        maxheap_array = faiss.float_maxheap_array_t()

        self.assertTrue("a template structure for a set of [min|max]-heaps"
                        in maxheap_array.__doc__)
