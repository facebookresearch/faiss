# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import faiss


class TestParameterSpace(unittest.TestCase):

    def test_nprobe(self):
        index = faiss.index_factory(32, "IVF32,Flat")
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", 5)
        self.assertEqual(index.nprobe, 5)

    def test_nprobe_2(self):
        index = faiss.index_factory(32, "IDMap,IVF32,Flat")
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", 5)
        index2 = faiss.downcast_index(index.index)
        self.assertEqual(index2.nprobe, 5)

    def test_nprobe_3(self):
        index = faiss.index_factory(32, "IVF32,SQ8,RFlat")
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", 5)
        index2 = faiss.downcast_index(index.base_index)
        self.assertEqual(index2.nprobe, 5)

    def test_nprobe_4(self):
        index = faiss.index_factory(32, "PCAR32,IVF32,SQ8,RFlat")
        ps = faiss.ParameterSpace()

        ps.set_index_parameter(index, "nprobe", 5)
        index2 = faiss.downcast_index(index.base_index)
        index2 = faiss.downcast_index(index2.index)
        self.assertEqual(index2.nprobe, 5)

    def test_efSearch(self):
        index = faiss.index_factory(32, "IVF32_HNSW32,SQ8")
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "quantizer_efSearch", 5)
        index2 = faiss.downcast_index(index.quantizer)
        self.assertEqual(index2.hnsw.efSearch, 5)
