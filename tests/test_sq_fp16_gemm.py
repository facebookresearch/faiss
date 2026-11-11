# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Python-level checks for IndexScalarQuantizer.use_fp16_gemm."""

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


def enable_gemm(index, kernel=None, bs_x=None, bs_y=512, rescore=None):
    index.use_fp16_gemm = True
    if kernel is not None:
        index.fp16_gemm_kernel = kernel
    if bs_x is not None:
        index.fp16_gemm_bs_x = bs_x
    index.fp16_gemm_bs_y = bs_y
    if rescore is not None:
        index.fp16_gemm_rescore = rescore
    return index


class TestSQFP16Gemm(unittest.TestCase):

    def build(self, metric, xb, kernel=None):
        index = faiss.IndexScalarQuantizer(
            xb.shape[1], faiss.ScalarQuantizer.QT_fp16, metric)
        enable_gemm(index, kernel)
        index.add(xb)
        return index

    def check_matches_native(self, metric):
        xb, xq = make_data()
        ref = faiss.IndexScalarQuantizer(
            xb.shape[1], faiss.ScalarQuantizer.QT_fp16, metric)
        ref.add(xb)
        # The gemm path rounds queries to fp16 before scoring.
        xq_rounded = xq.astype("float16").astype("float32")
        Dref, Iref = ref.search(xq_rounded, 10)
        for kernel in available_kernels():
            index = self.build(metric, xb, kernel)
            self.assertEqual(index.fp16_gemm_effective_kernel(), kernel)
            D, I = index.search(xq, 10)
            np.testing.assert_array_equal(I, Iref)
            # Same tolerance as the C++ tests: the native distance code may
            # or may not fuse multiply-adds depending on the SIMD build.
            tol = 1e-4 * np.maximum(1.0, np.abs(Dref))
            self.assertTrue(np.all(np.abs(D - Dref) <= tol))
            self.assertEqual(index.fp16_gemm_nb_queries(), xq.shape[0])

    def test_matches_native_ip(self):
        self.check_matches_native(faiss.METRIC_INNER_PRODUCT)

    def test_matches_native_l2(self):
        self.check_matches_native(faiss.METRIC_L2)

    def test_disabled_by_default(self):
        xb, xq = make_data(nb=500)
        index = faiss.IndexScalarQuantizer(
            xb.shape[1], faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
        self.assertFalse(index.use_fp16_gemm)
        index.add(xb)
        index.search(xq, 10)
        self.assertEqual(index.fp16_gemm_packed_bytes(), 0)
        self.assertEqual(index.fp16_gemm_nb_queries(), 0)

    def check_same_search(self, index, other, xq):
        self.assertEqual(other.ntotal, index.ntotal)
        D, I = index.search(xq, 10)
        D2, I2 = other.search(xq, 10)
        np.testing.assert_array_equal(I2, I)
        np.testing.assert_array_equal(D2, D)

    def test_serialize_and_clone(self):
        xb, xq = make_data()
        index = self.build(faiss.METRIC_L2, xb)
        index.fp16_gemm_bs_x = 64
        index.fp16_gemm_rescore = False

        clone = faiss.clone_index(index)
        self.assertTrue(clone.use_fp16_gemm)
        self.assertEqual(clone.fp16_gemm_kernel, index.fp16_gemm_kernel)
        self.assertEqual(clone.fp16_gemm_bs_x, index.fp16_gemm_bs_x)
        self.assertEqual(clone.fp16_gemm_bs_y, index.fp16_gemm_bs_y)
        self.assertEqual(clone.fp16_gemm_rescore, index.fp16_gemm_rescore)
        self.check_same_search(index, clone, xq)

        # The settings are runtime-only and are not serialized.
        roundtrip = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(roundtrip, faiss.IndexScalarQuantizer)
        self.assertFalse(roundtrip.use_fp16_gemm)
        enable_gemm(roundtrip, index.fp16_gemm_kernel, bs_x=64, rescore=False)
        self.check_same_search(index, roundtrip, xq)

    def test_ivf_coarse_quantizer(self):
        xb, xq = make_data(nb=5000)
        d = xb.shape[1]
        quantizer = enable_gemm(faiss.IndexScalarQuantizer(
            d, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2))
        index = faiss.IndexIVFFlat(quantizer, d, 64)
        index.train(xb)
        index.add(xb)
        index.nprobe = 8
        D, I = index.search(xq, 10)
        self.assertGreater(quantizer.fp16_gemm_nb_queries(), 0)

        roundtrip = faiss.deserialize_index(faiss.serialize_index(index))
        q2 = faiss.downcast_index(roundtrip.quantizer)
        self.assertIsInstance(q2, faiss.IndexScalarQuantizer)
        enable_gemm(q2)
        roundtrip.nprobe = 8
        D2, I2 = roundtrip.search(xq, 10)
        np.testing.assert_array_equal(I2, I)
        np.testing.assert_array_equal(D2, D)
        self.assertGreater(q2.fp16_gemm_nb_queries(), 0)

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
