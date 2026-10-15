# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np


def rowwise_int8_encode(x):
    """Reference encoder: s = max|x| / 127 (1 for a zero row),
    c = round-half-even(x / s) clipped to [-127, 127]."""
    x = np.asarray(x, dtype="float32")
    s = np.abs(x).max(1) / np.float32(127.0)
    s[s == 0] = 1.0
    c = np.clip(np.rint(x / s[:, None]), -127, 127).astype("int8")
    return c, s.astype("float32")


def reference_search(cq, sq, cb, sb, k, metric):
    """Exact search on the int8 codes (int64 dot products, float64)."""
    dot = cq.astype("int64") @ cb.astype("int64").T
    ip = (sq.astype("float64")[:, None] * sb.astype("float64")[None, :]) * dot
    if metric == faiss.METRIC_L2:
        nq = sq.astype("float64") ** 2 * (cq.astype("int64") ** 2).sum(1)
        nb = sb.astype("float64") ** 2 * (cb.astype("int64") ** 2).sum(1)
        dis = nq[:, None] + nb[None, :] - 2 * ip
        idx = np.argsort(dis, axis=1, kind="stable")[:, :k]
    else:
        dis = ip
        idx = np.argsort(-dis, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(dis, idx, 1), idx


def i8ptr(a):
    return faiss.cast_integer_to_int8_ptr(a.ctypes.data)


def get_int8(index):
    codes = np.empty((index.ntotal, index.d), dtype="int8")
    scales = np.empty(index.ntotal, dtype="float32")
    index.get_int8(0, index.ntotal, i8ptr(codes), faiss.swig_ptr(scales))
    return codes, scales


def search_int8(index, codes, scales, k):
    codes = np.ascontiguousarray(codes, dtype="int8")
    scales = np.ascontiguousarray(scales, dtype="float32")
    n = codes.shape[0]
    D = np.empty((n, k), dtype="float32")
    I = np.empty((n, k), dtype="int64")
    index.search_int8(
        n, i8ptr(codes), faiss.swig_ptr(scales), k,
        faiss.swig_ptr(D), faiss.swig_ptr(I))
    return D, I


class TestIndexRowwiseInt8(unittest.TestCase):

    def setUp(self):
        rs = np.random.RandomState(123)
        self.d = 128
        # rows with very different norms, plus one all-zero row
        self.xb = (rs.randn(3000, self.d)
                   * rs.uniform(0.1, 3, (3000, 1))).astype("float32")
        self.xb[7] = 0
        self.xq = (rs.randn(100, self.d)
                   * rs.uniform(0.1, 3, (100, 1))).astype("float32")

    def check_search(self, metric):
        k = 5
        index = faiss.IndexRowwiseInt8(self.d, metric)
        index.add(self.xb)
        cb, sb = rowwise_int8_encode(self.xb)
        cq, sq = rowwise_int8_encode(self.xq)

        # encoding is bit-identical to the reference
        c2, s2 = get_int8(index)
        np.testing.assert_array_equal(c2, cb)
        np.testing.assert_array_equal(s2, sb)

        D, I = index.search(self.xq, k)
        Dref, Iref = reference_search(cq, sq, cb, sb, k, metric)
        np.testing.assert_array_equal(I, Iref)
        np.testing.assert_allclose(
            D, Dref, rtol=1e-5, atol=1e-5 * np.abs(Dref).max())

        # int8 queries give the same result as float queries
        D8, I8 = search_int8(index, cq, sq, k)
        np.testing.assert_array_equal(D8, D)
        np.testing.assert_array_equal(I8, I)

    def test_search_L2(self):
        self.check_search(faiss.METRIC_L2)

    def test_search_IP(self):
        self.check_search(faiss.METRIC_INNER_PRODUCT)

    def test_add_int8(self):
        cb, sb = rowwise_int8_encode(self.xb)
        a = faiss.IndexRowwiseInt8(self.d)
        a.add(self.xb)
        b = faiss.IndexRowwiseInt8(self.d)
        b.add_int8(len(cb), i8ptr(cb), faiss.swig_ptr(sb))
        self.assertEqual(b.ntotal, a.ntotal)
        np.testing.assert_array_equal(
            faiss.vector_to_array(a.codes), faiss.vector_to_array(b.codes))

    def test_reconstruct(self):
        index = faiss.IndexRowwiseInt8(self.d)
        index.add(self.xb)
        cb, sb = rowwise_int8_encode(self.xb)
        np.testing.assert_array_equal(
            index.reconstruct_n(0, index.ntotal),
            cb.astype("float32") * sb[:, None])
        # quantization error is at most half a step per component
        err = np.abs(index.reconstruct_n(0, index.ntotal) - self.xb)
        self.assertTrue(np.all(err <= sb[:, None] / 2 + 1e-6))

    def test_serialize(self):
        index = faiss.IndexRowwiseInt8(self.d, faiss.METRIC_INNER_PRODUCT)
        index.add(self.xb)
        index2 = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(index2, faiss.IndexRowwiseInt8)
        self.assertEqual(index2.metric_type, faiss.METRIC_INNER_PRODUCT)
        self.assertEqual(index2.code_size, self.d + 4)
        np.testing.assert_array_equal(
            faiss.vector_to_array(index.codes),
            faiss.vector_to_array(index2.codes))
        Da, Ia = index.search(self.xq, 4)
        Db, Ib = index2.search(self.xq, 4)
        np.testing.assert_array_equal(Ia, Ib)
        np.testing.assert_array_equal(Da, Db)

    def test_distance_computer(self):
        index = faiss.IndexRowwiseInt8(self.d)
        index.add(self.xb)
        D, I = index.search(self.xq[:5], 3)
        dc = index.get_distance_computer()
        for qi in range(5):
            dc.set_query(faiss.swig_ptr(self.xq[qi]))
            for r in range(3):
                self.assertAlmostEqual(
                    dc(int(I[qi, r])), float(D[qi, r]),
                    delta=1e-4 * abs(float(D[qi, r])) + 1e-4)

    def test_k_larger_than_ntotal(self):
        index = faiss.IndexRowwiseInt8(self.d)
        index.add(self.xb[:3])
        D, I = index.search(self.xq[:4], 5)
        self.assertTrue(np.all(I[:, :3] >= 0))
        self.assertTrue(np.all(I[:, 3:] == -1))

    def test_dimension_not_multiple_of_64(self):
        # exercises the scalar (non-VNNI) dot product path
        d = 50
        xb, xq = self.xb[:500, :d].copy(), self.xq[:20, :d].copy()
        index = faiss.IndexRowwiseInt8(d)
        index.add(xb)
        cb, sb = rowwise_int8_encode(xb)
        cq, sq = rowwise_int8_encode(xq)
        D, I = index.search(xq, 4)
        _, Iref = reference_search(cq, sq, cb, sb, 4, faiss.METRIC_L2)
        np.testing.assert_array_equal(I, Iref)


if __name__ == "__main__":
    unittest.main()
