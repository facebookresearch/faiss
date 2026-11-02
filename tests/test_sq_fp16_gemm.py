# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Python-level checks for IndexSQFP16Gemm: results, I/O, clone, kernels."""

import unittest

import faiss
import numpy as np


def available_kernels():
    kernels = [faiss.GemmKernel_SCALAR]
    if faiss.fma_available():
        kernels.append(faiss.GemmKernel_FMA)
    if faiss.amx_fp16_available():
        kernels.append(faiss.GemmKernel_AMX)
    return kernels


def make_data(nb=3000, nq=97, d=72, seed=123):
    rs = np.random.RandomState(seed)
    xb = rs.standard_normal((nb, d)).astype("float32")
    xq = rs.standard_normal((nq, d)).astype("float32")
    return xb, xq


class TestSQFP16Gemm(unittest.TestCase):

    def build(self, metric, xb, kernel=None):
        index = faiss.IndexSQFP16Gemm(xb.shape[1], metric)
        if kernel is not None:
            index.kernel = kernel
        index.bs_y = 512
        index.add(xb)
        return index

    def check_matches_native(self, metric):
        xb, xq = make_data()
        ref = faiss.IndexScalarQuantizer(
            xb.shape[1], faiss.ScalarQuantizer.QT_fp16, metric)
        ref.add(xb)
        # IndexSQFP16Gemm rounds queries to fp16 before scoring.
        xq_rounded = xq.astype("float16").astype("float32")
        Dref, Iref = ref.search(xq_rounded, 10)
        for kernel in available_kernels():
            index = self.build(metric, xb, kernel)
            self.assertEqual(index.effective_kernel(), kernel)
            D, I = index.search(xq, 10)
            np.testing.assert_array_equal(I, Iref)
            # Same tolerance as the C++ tests: the native distance code may
            # or may not fuse multiply-adds depending on the SIMD build.
            tol = 1e-4 * np.maximum(1.0, np.abs(Dref))
            self.assertTrue(np.all(np.abs(D - Dref) <= tol))

    def test_matches_native_ip(self):
        self.check_matches_native(faiss.METRIC_INNER_PRODUCT)

    def test_matches_native_l2(self):
        self.check_matches_native(faiss.METRIC_L2)

    def check_same_index(self, index, other, xq):
        self.assertIsInstance(other, faiss.IndexSQFP16Gemm)
        self.assertEqual(other.kernel, index.kernel)
        self.assertEqual(other.use_gemm, index.use_gemm)
        self.assertEqual(other.bs_x, index.bs_x)
        self.assertEqual(other.bs_y, index.bs_y)
        self.assertEqual(other.rescore_native, index.rescore_native)
        self.assertEqual(other.ntotal, index.ntotal)
        D, I = index.search(xq, 10)
        D2, I2 = other.search(xq, 10)
        np.testing.assert_array_equal(I2, I)
        np.testing.assert_array_equal(D2, D)

    def test_serialize_and_clone(self):
        xb, xq = make_data()
        index = self.build(faiss.METRIC_L2, xb)
        index.bs_x = 64
        index.rescore_native = False
        roundtrip = faiss.deserialize_index(faiss.serialize_index(index))
        self.check_same_index(index, roundtrip, xq)
        self.check_same_index(index, faiss.clone_index(index), xq)

    def test_ivf_coarse_quantizer(self):
        xb, xq = make_data(nb=5000)
        d = xb.shape[1]
        quantizer = faiss.IndexSQFP16Gemm(d, faiss.METRIC_L2)
        index = faiss.IndexIVFFlat(quantizer, d, 64)
        index.train(xb)
        index.add(xb)
        index.nprobe = 8
        D, I = index.search(xq, 10)

        roundtrip = faiss.deserialize_index(faiss.serialize_index(index))
        q2 = faiss.downcast_index(roundtrip.quantizer)
        self.assertIsInstance(q2, faiss.IndexSQFP16Gemm)
        roundtrip.nprobe = 8
        D2, I2 = roundtrip.search(xq, 10)
        np.testing.assert_array_equal(I2, I)
        np.testing.assert_array_equal(D2, D)

    def test_unavailable_kernel_throws(self):
        xb, xq = make_data(nb=100, nq=4)
        for kernel, available in [
            (faiss.GemmKernel_AMX, faiss.amx_fp16_available()),
            (faiss.GemmKernel_FMA, faiss.fma_available()),
        ]:
            if available:
                continue
            index = self.build(faiss.METRIC_L2, xb, kernel)
            with self.assertRaises(RuntimeError):
                index.search(xq, 5)


if __name__ == "__main__":
    unittest.main()
