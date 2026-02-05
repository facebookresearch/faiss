# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import faiss


class TestBinaryParameterSpace(unittest.TestCase):

    def test_nprobe(self):
        d = 64
        quantizer = faiss.IndexBinaryFlat(d)
        index = faiss.IndexBinaryIVF(quantizer, d, 32)
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", 5)
        self.assertEqual(index.nprobe, 5)

    def test_nprobe_2(self):
        d = 64
        quantizer = faiss.IndexBinaryFlat(d)
        index_ivf = faiss.IndexBinaryIVF(quantizer, d, 32)
        index = faiss.IndexBinaryIDMap(index_ivf)
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", 5)
        index2 = faiss.downcast_IndexBinary(index.index)
        self.assertEqual(index2.nprobe, 5)

    def test_efSearch(self):
        d = 64
        index = faiss.IndexBinaryHNSW(d, 32)
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "efSearch", 10)
        self.assertEqual(index.hnsw.efSearch, 10)

    def test_efConstruction(self):
        d = 64
        index = faiss.IndexBinaryHNSW(d, 32)
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "efConstruction", 40)
        self.assertEqual(index.hnsw.efConstruction, 40)

    def test_quantizer_efSearch(self):
        d = 64
        quantizer = faiss.IndexBinaryHNSW(d, 32)
        index = faiss.IndexBinaryIVF(quantizer, d, 32)
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "quantizer_efSearch", 8)
        quantizer2 = faiss.downcast_IndexBinary(index.quantizer)
        self.assertEqual(quantizer2.hnsw.efSearch, 8)

    def test_quantizer_efConstruction(self):
        d = 64
        quantizer = faiss.IndexBinaryHNSW(d, 32)
        index = faiss.IndexBinaryIVF(quantizer, d, 32)
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "quantizer_efConstruction", 50)
        quantizer2 = faiss.downcast_IndexBinary(index.quantizer)
        self.assertEqual(quantizer2.hnsw.efConstruction, 50)
