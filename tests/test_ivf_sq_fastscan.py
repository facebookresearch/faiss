#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest

import faiss
from faiss.contrib.datasets import SyntheticDataset


class TestIndexIVFSQFastScanRecall(unittest.TestCase):

    def _compare_recall(self, qtype, nlist=16, nprobe=16):
        ds = SyntheticDataset(d=32, nt=2000, nb=2000, nq=50, seed=42)
        k = 10

        gt_idx = faiss.IndexFlatL2(ds.d)
        gt_idx.add(ds.get_database())
        D_gt, I_gt = gt_idx.search(ds.get_queries(), k)

        # Shared coarse quantizer ensures same IVF assignment
        quantizer = faiss.IndexFlatL2(ds.d)

        ref = faiss.IndexIVFScalarQuantizer(
            quantizer, ds.d, nlist, qtype
        )
        ref.train(ds.get_train())

        test = faiss.IndexIVFSQFastScan(
            quantizer, ds.d, nlist, qtype
        )
        test.sq = ref.sq
        test.is_trained = True
        test.add(ds.get_database())
        self.assertEqual(test.ntotal, ds.nb)
        test.nprobe = nprobe
        D_test, I_test = test.search(ds.get_queries(), k)

        recall_test = np.mean([
            len(set(I_test[i]) & set(I_gt[i])) / k
            for i in range(ds.nq)
        ])

        return recall_test

    def test_recall_4bit(self):
        recall_test = self._compare_recall(
            faiss.ScalarQuantizer.QT_4bit
        )
        self.assertGreater(recall_test, 0.5)

    def test_recall_8bit(self):
        recall_test = self._compare_recall(
            faiss.ScalarQuantizer.QT_8bit
        )
        self.assertGreater(recall_test, 0.7)

    def test_recall_6bit(self):
        recall_test = self._compare_recall(
            faiss.ScalarQuantizer.QT_6bit
        )
        self.assertGreater(recall_test, 0.6)

    def test_recall_fp16(self):
        recall_test = self._compare_recall(
            faiss.ScalarQuantizer.QT_fp16
        )
        self.assertGreater(recall_test, 0.9)


class TestIndexIVFSQFastScanFactory(unittest.TestCase):

    def test_factory_strings(self):
        d = 32
        for s in ("IVF16,SQ4fs", "IVF16,SQ6fs", "IVF16,SQfp16fs"):
            with self.subTest(s=s):
                idx = faiss.index_factory(d, s)
                self.assertIsInstance(idx, faiss.IndexIVFSQFastScan)

    def test_factory_with_bbs(self):
        d = 32
        idx = faiss.index_factory(d, "IVF16,SQ8fs_64")
        self.assertIsInstance(idx, faiss.IndexIVFSQFastScan)
        self.assertEqual(idx.bbs, 64)

    def test_factory_train_search(self):
        ds = SyntheticDataset(d=32, nt=1000, nb=1000, nq=10, seed=42)

        idx = faiss.index_factory(ds.d, "IVF16,SQ8fs")
        idx.train(ds.get_train())
        idx.add(ds.get_database())
        idx.nprobe = 16
        D, I = idx.search(ds.get_queries(), 5)
        self.assertTrue(np.all(I >= 0))


class TestIndexIVFSQFastScanIP(unittest.TestCase):

    def test_inner_product(self):
        ds = SyntheticDataset(d=32, nt=1000, nb=1000, nq=10, seed=42)

        for qtype in (faiss.ScalarQuantizer.QT_4bit,
                      faiss.ScalarQuantizer.QT_8bit):
            with self.subTest(qtype=qtype):
                quantizer = faiss.IndexFlatIP(ds.d)
                idx = faiss.IndexIVFSQFastScan(
                    quantizer, ds.d, 8, qtype, faiss.METRIC_INNER_PRODUCT
                )
                idx.train(ds.get_train())
                idx.add(ds.get_database())
                idx.nprobe = 8
                D, I = idx.search(ds.get_queries(), 10)
                self.assertTrue(np.all(I >= 0))


class TestIndexIVFSQFastScanIO(unittest.TestCase):

    def test_io_roundtrip(self):
        ds = SyntheticDataset(d=32, nt=500, nb=500, nq=5, seed=42)

        for qtype in (faiss.ScalarQuantizer.QT_4bit,
                      faiss.ScalarQuantizer.QT_8bit):
            with self.subTest(qtype=qtype):
                quantizer = faiss.IndexFlatL2(ds.d)
                idx = faiss.IndexIVFSQFastScan(quantizer, ds.d, 8, qtype)
                idx.train(ds.get_train())
                idx.add(ds.get_database())
                idx.nprobe = 8

                D1, I1 = idx.search(ds.get_queries(), 5)

                idx2 = faiss.deserialize_index(faiss.serialize_index(idx))
                idx2.nprobe = 8
                D2, I2 = idx2.search(ds.get_queries(), 5)
                np.testing.assert_array_equal(I1, I2)
                np.testing.assert_allclose(D1, D2, rtol=1e-5)


class TestIndexIVFSQFastScanReset(unittest.TestCase):

    def test_reset(self):
        ds = SyntheticDataset(d=32, nt=500, nb=500, nq=0, seed=42)

        quantizer = faiss.IndexFlatL2(ds.d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, ds.d, 8, faiss.ScalarQuantizer.QT_8bit
        )
        idx.train(ds.get_train())
        idx.add(ds.get_database())

        idx.reset()
        self.assertEqual(idx.ntotal, 0)

        idx.add(ds.get_database())
        self.assertEqual(idx.ntotal, ds.nb)


class TestIndexIVFSQFastScanReconstruct(unittest.TestCase):

    def test_reconstruct(self):
        ds = SyntheticDataset(d=32, nt=200, nb=200, nq=0, seed=42)

        for qtype, tol in ((faiss.ScalarQuantizer.QT_4bit, 5.0),
                           (faiss.ScalarQuantizer.QT_8bit, 1.0)):
            with self.subTest(qtype=qtype):
                quantizer = faiss.IndexFlatL2(ds.d)
                idx = faiss.IndexIVFSQFastScan(quantizer, ds.d, 8, qtype)
                idx.train(ds.get_train())
                idx.add(ds.get_database())
                idx.make_direct_map()

                recon = idx.reconstruct(0)
                err = np.linalg.norm(recon - ds.get_database()[0])
                self.assertLess(err, tol)


class TestIndexIVFSQFastScanEdgeCases(unittest.TestCase):

    def test_single_vector(self):
        ds = SyntheticDataset(d=32, nt=100, nb=1, nq=1, seed=42)

        quantizer = faiss.IndexFlatL2(ds.d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, ds.d, 4, faiss.ScalarQuantizer.QT_4bit
        )
        idx.train(ds.get_train())
        idx.add(ds.get_database())
        idx.nprobe = 4
        D, I = idx.search(ds.get_queries(), 1)
        self.assertEqual(I[0, 0], 0)

    def test_odd_dimension(self):
        ds = SyntheticDataset(d=33, nt=500, nb=500, nq=10, seed=42)

        quantizer = faiss.IndexFlatL2(ds.d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, ds.d, 8, faiss.ScalarQuantizer.QT_4bit
        )
        idx.train(ds.get_train())
        idx.add(ds.get_database())
        idx.nprobe = 8
        D, I = idx.search(ds.get_queries(), 5)
        self.assertTrue(np.all(I >= 0))


if __name__ == "__main__":
    unittest.main()
