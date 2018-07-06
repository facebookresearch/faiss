# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

import unittest
import faiss


class TestBinaryFactory(unittest.TestCase):

    def test_factory_1(self):

        index = faiss.index_binary_factory(16, "BIVF10")
        assert index.invlists is not None

    def test_factory_2(self):

        index = faiss.index_binary_factory(16, "BFlat")
        assert index.code_size == 2
