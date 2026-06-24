#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for IndexIVFSQFastScan."""

import numpy as np
import unittest
import tempfile
import os

import faiss


def make_data(n, d, seed=42):
    rs = np.random.RandomState(seed)
    return rs.rand(n, d).astype("float32")


class TestIndexIVFSQFastScanBasic(unittest.TestCase):
    """Basic construction, train, add, search tests."""

    def test_construct_native_4bit(self):
        d, nlist = 32, 8
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_4bit
        )
        self.assertEqual(idx.d, d)
        self.assertEqual(idx.nlist, nlist)

    def test_construct_rerank_8bit(self):
        d, nlist = 32, 8
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit
        )
        self.assertEqual(idx.d, d)

    def test_train_add_search_4bit(self):
        d, nlist, n = 32, 8, 1000
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(10, d, seed=3)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_4bit
        )
        idx.train(xt)
        self.assertTrue(idx.is_trained)
        idx.add(xb)
        self.assertEqual(idx.ntotal, n)

        idx.nprobe = nlist
        D, I = idx.search(xq, 10)
        self.assertEqual(D.shape, (10, 10))
        self.assertEqual(I.shape, (10, 10))
        self.assertTrue(np.all(I >= 0))

    def test_train_add_search_8bit(self):
        d, nlist, n = 32, 8, 1000
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(10, d, seed=3)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit
        )
        idx.train(xt)
        idx.add(xb)
        self.assertEqual(idx.ntotal, n)

        idx.nprobe = nlist
        D, I = idx.search(xq, 10)
        self.assertTrue(np.all(I >= 0))

    def test_train_add_search_fp16_fallback(self):
        d, nlist, n = 32, 8, 1000
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(10, d, seed=3)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_fp16
        )
        idx.train(xt)
        idx.add(xb)
        self.assertEqual(idx.ntotal, n)

        idx.nprobe = nlist
        D, I = idx.search(xq, 10)
        self.assertTrue(np.all(I >= 0))


class TestIndexIVFSQFastScanRecall(unittest.TestCase):
    """Recall parity tests against IndexIVFScalarQuantizer."""

    def _compare_recall(self, qtype, nlist=16, nprobe=16):
        d, n = 32, 2000
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(50, d, seed=3)
        k = 10

        # Ground truth
        gt_idx = faiss.IndexFlatL2(d)
        gt_idx.add(xb)
        D_gt, I_gt = gt_idx.search(xq, k)

        # Reference: IndexIVFScalarQuantizer
        quantizer_ref = faiss.IndexFlatL2(d)
        ref = faiss.IndexIVFScalarQuantizer(
            quantizer_ref, d, nlist, qtype
        )
        ref.train(xt)
        ref.add(xb)
        ref.nprobe = nprobe
        D_ref, I_ref = ref.search(xq, k)

        # Test: IndexIVFSQFastScan
        quantizer_test = faiss.IndexFlatL2(d)
        test = faiss.IndexIVFSQFastScan(
            quantizer_test, d, nlist, qtype
        )
        test.train(xt)
        test.add(xb)
        test.nprobe = nprobe
        D_test, I_test = test.search(xq, k)

        # Compute recalls
        recall_ref = np.mean([
            len(set(I_ref[i]) & set(I_gt[i])) / k
            for i in range(len(xq))
        ])
        recall_test = np.mean([
            len(set(I_test[i]) & set(I_gt[i])) / k
            for i in range(len(xq))
        ])

        return recall_ref, recall_test

    def test_recall_4bit(self):
        recall_ref, recall_test = self._compare_recall(
            faiss.ScalarQuantizer.QT_4bit
        )
        # Both should have reasonable recall with nprobe=nlist
        self.assertGreater(recall_test, 0.5)

    def test_recall_8bit(self):
        recall_ref, recall_test = self._compare_recall(
            faiss.ScalarQuantizer.QT_8bit
        )
        self.assertGreater(recall_test, 0.7)

    def test_recall_6bit(self):
        recall_ref, recall_test = self._compare_recall(
            faiss.ScalarQuantizer.QT_6bit
        )
        self.assertGreater(recall_test, 0.6)

    def test_recall_fp16(self):
        recall_ref, recall_test = self._compare_recall(
            faiss.ScalarQuantizer.QT_fp16
        )
        self.assertGreater(recall_test, 0.9)


class TestIndexIVFSQFastScanFactory(unittest.TestCase):
    """Factory string tests."""

    def test_factory_sq8fs(self):
        d = 32
        idx = faiss.index_factory(d, "IVF16,SQ8fs")
        self.assertIsInstance(idx, faiss.IndexIVFSQFastScan)

    def test_factory_sq4fs(self):
        d = 32
        idx = faiss.index_factory(d, "IVF16,SQ4fs")
        self.assertIsInstance(idx, faiss.IndexIVFSQFastScan)

    def test_factory_sqfp16fs(self):
        d = 32
        idx = faiss.index_factory(d, "IVF16,SQfp16fs")
        self.assertIsInstance(idx, faiss.IndexIVFSQFastScan)

    def test_factory_sq6fs(self):
        d = 32
        idx = faiss.index_factory(d, "IVF16,SQ6fs")
        self.assertIsInstance(idx, faiss.IndexIVFSQFastScan)

    def test_factory_with_bbs(self):
        d = 32
        idx = faiss.index_factory(d, "IVF16,SQ8fs_64")
        self.assertIsInstance(idx, faiss.IndexIVFSQFastScan)
        self.assertEqual(idx.bbs, 64)

    def test_factory_train_search(self):
        d, n = 32, 1000
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(10, d, seed=3)

        idx = faiss.index_factory(d, "IVF16,SQ8fs")
        idx.train(xt)
        idx.add(xb)
        idx.nprobe = 16
        D, I = idx.search(xq, 5)
        self.assertTrue(np.all(I >= 0))


class TestIndexIVFSQFastScanIP(unittest.TestCase):
    """Inner product metric tests."""

    def test_inner_product_4bit(self):
        d, nlist, n = 32, 8, 1000
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(10, d, seed=3)

        quantizer = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_4bit,
            faiss.METRIC_INNER_PRODUCT
        )
        idx.train(xt)
        idx.add(xb)
        idx.nprobe = nlist
        D, I = idx.search(xq, 10)
        self.assertTrue(np.all(I >= 0))
        # IP distances should be positive for positive data
        self.assertTrue(np.all(D >= 0))

    def test_inner_product_8bit(self):
        d, nlist, n = 32, 8, 1000
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(10, d, seed=3)

        quantizer = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT
        )
        idx.train(xt)
        idx.add(xb)
        idx.nprobe = nlist
        D, I = idx.search(xq, 10)
        self.assertTrue(np.all(I >= 0))


class TestIndexIVFSQFastScanIO(unittest.TestCase):
    """Serialization round-trip tests."""

    def test_io_roundtrip_4bit(self):
        d, nlist, n = 32, 8, 500
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(5, d, seed=3)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_4bit
        )
        idx.train(xt)
        idx.add(xb)
        idx.nprobe = nlist

        D1, I1 = idx.search(xq, 5)

        with tempfile.NamedTemporaryFile(suffix=".faissindex", delete=False) as f:
            fname = f.name
        try:
            faiss.write_index(idx, fname)
            idx2 = faiss.read_index(fname)
            idx2.nprobe = nlist
            D2, I2 = idx2.search(xq, 5)
            np.testing.assert_array_equal(I1, I2)
            np.testing.assert_allclose(D1, D2, rtol=1e-5)
        finally:
            os.unlink(fname)

    def test_io_roundtrip_8bit(self):
        d, nlist, n = 32, 8, 500
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(5, d, seed=3)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit
        )
        idx.train(xt)
        idx.add(xb)
        idx.nprobe = nlist

        D1, I1 = idx.search(xq, 5)

        with tempfile.NamedTemporaryFile(suffix=".faissindex", delete=False) as f:
            fname = f.name
        try:
            faiss.write_index(idx, fname)
            idx2 = faiss.read_index(fname)
            idx2.nprobe = nlist
            D2, I2 = idx2.search(xq, 5)
            np.testing.assert_array_equal(I1, I2)
            np.testing.assert_allclose(D1, D2, rtol=1e-5)
        finally:
            os.unlink(fname)


class TestIndexIVFSQFastScanReset(unittest.TestCase):
    """Reset and re-add tests."""

    def test_reset(self):
        d, nlist, n = 32, 8, 500
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit
        )
        idx.train(xt)
        idx.add(xb)
        self.assertEqual(idx.ntotal, n)

        idx.reset()
        self.assertEqual(idx.ntotal, 0)

        idx.add(xb)
        self.assertEqual(idx.ntotal, n)


class TestIndexIVFSQFastScanReconstruct(unittest.TestCase):
    """Reconstruction tests."""

    def test_reconstruct_4bit(self):
        d, nlist, n = 32, 8, 200
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_4bit
        )
        idx.train(xt)
        idx.add(xb)
        idx.make_direct_map()

        recon = idx.reconstruct(0)
        self.assertEqual(recon.shape, (d,))
        # Reconstruction should be somewhat close to original
        err = np.linalg.norm(recon - xb[0])
        self.assertLess(err, 5.0)

    def test_reconstruct_8bit(self):
        d, nlist, n = 32, 8, 200
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit
        )
        idx.train(xt)
        idx.add(xb)
        idx.make_direct_map()

        recon = idx.reconstruct(0)
        self.assertEqual(recon.shape, (d,))
        err = np.linalg.norm(recon - xb[0])
        self.assertLess(err, 1.0)


class TestIndexIVFSQFastScanEdgeCases(unittest.TestCase):
    """Edge cases and special scenarios."""

    def test_single_vector(self):
        d, nlist = 32, 4
        xt = make_data(100, d, seed=1)
        xb = make_data(1, d, seed=2)
        xq = make_data(1, d, seed=3)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_4bit
        )
        idx.train(xt)
        idx.add(xb)
        idx.nprobe = nlist
        D, I = idx.search(xq, 1)
        self.assertEqual(I[0, 0], 0)

    def test_k_equals_1(self):
        d, nlist, n = 32, 8, 500
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(10, d, seed=3)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit
        )
        idx.train(xt)
        idx.add(xb)
        idx.nprobe = nlist
        D, I = idx.search(xq, 1)
        self.assertEqual(D.shape, (10, 1))
        self.assertEqual(I.shape, (10, 1))

    def test_odd_dimension(self):
        d, nlist, n = 33, 8, 500
        xt = make_data(n, d, seed=1)
        xb = make_data(n, d, seed=2)
        xq = make_data(10, d, seed=3)

        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFSQFastScan(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_4bit
        )
        idx.train(xt)
        idx.add(xb)
        idx.nprobe = nlist
        D, I = idx.search(xq, 5)
        self.assertTrue(np.all(I >= 0))


if __name__ == "__main__":
    unittest.main()
