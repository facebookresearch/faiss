# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np


class TestFp16LinearTransform(unittest.TestCase):
    def require_fp16(self):
        probe = faiss.LinearTransform()
        if not probe.fp16_supported():
            self.skipTest("FP16 linear transforms are unavailable")

    def make_transform(self, d_in, d_out, bias, seed):
        rs = np.random.RandomState(seed)
        matrix = (rs.randn(d_out, d_in) / np.sqrt(d_in)).astype("float32")
        transform = faiss.LinearTransform(d_in, d_out, bias is not None)
        faiss.copy_array_to_vector(matrix.ravel(), transform.A)
        if bias is not None:
            faiss.copy_array_to_vector(bias, transform.b)
        transform.is_trained = True
        return transform, matrix

    def test_scalar_oracle_tails_zero_and_bias(self):
        self.require_fp16()
        for d_in in (1, 7, 8, 15, 16, 17, 31, 32, 33, 65, 128, 768, 1537):
            d_out = 5
            rs = np.random.RandomState(1000 + d_in)
            bias = rs.randn(d_out).astype("float32")
            transform, matrix = self.make_transform(d_in, d_out, bias, d_in)
            transform.prepare_fp16()
            queries = rs.randn(3, d_in).astype("float32")
            queries[0] = 0
            actual = np.empty((len(queries), d_out), dtype="float32")
            transform.apply_noalloc_fp16(
                len(queries), faiss.swig_ptr(queries), faiss.swig_ptr(actual)
            )

            half_matrix = matrix.astype("float16").astype("float32")
            half_queries = queries.astype("float16").astype("float32")
            expected = np.empty_like(actual)
            for row in range(len(queries)):
                for output in range(d_out):
                    value = np.float32(0)
                    for column in range(d_in):
                        value = np.float32(
                            value
                            + np.float32(
                                half_matrix[output, column]
                                * half_queries[row, column]
                            )
                        )
                    expected[row, output] = value + bias[output]
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_cache_is_explicit(self):
        self.require_fp16()
        transform, _ = self.make_transform(17, 9, None, 123)
        query = np.zeros(17, dtype="float32")
        output = np.empty(9, dtype="float32")
        with self.assertRaisesRegex(RuntimeError, "cache is missing or stale"):
            transform.apply_noalloc_fp16(
                1, faiss.swig_ptr(query), faiss.swig_ptr(output)
            )

    def test_batched_search_retains_fp32_path(self):
        rs = np.random.RandomState(99)
        d = 65
        xb = rs.randn(200, d).astype("float32")
        xq = rs.randn(3, d).astype("float32")
        rotation = faiss.RandomRotationMatrix(d, d)
        rotation.init(1234)
        storage = faiss.IndexFlatL2(d)
        index = faiss.IndexPreTransform(rotation, storage)
        index.own_fields = False
        index.add(xb)

        reference_distances, reference_ids = index.search(xq, 10)
        params = faiss.SearchParametersPreTransform()
        params.use_fp16_transform = True
        actual_distances, actual_ids = index.search(xq, 10, params=params)
        np.testing.assert_array_equal(actual_ids, reference_ids)
        np.testing.assert_array_equal(actual_distances, reference_distances)

    def test_single_query_search_uses_fp16_path(self):
        self.require_fp16()
        rs = np.random.RandomState(101)
        d = 65
        xb = rs.randn(200, d).astype("float32")
        xq = rs.randn(1, d).astype("float32")
        rotation = faiss.RandomRotationMatrix(d, d)
        rotation.init(1234)
        storage = faiss.IndexFlatL2(d)
        index = faiss.IndexPreTransform(rotation, storage)
        index.own_fields = False
        index.add(xb)

        rotation.prepare_fp16()
        transformed = np.empty_like(xq)
        rotation.apply_noalloc_fp16(
            1, faiss.swig_ptr(xq), faiss.swig_ptr(transformed)
        )
        expected_distances, expected_ids = storage.search(transformed, 10)

        params = faiss.SearchParametersPreTransform()
        params.use_fp16_transform = True
        actual_distances, actual_ids = index.search(xq, 10, params=params)
        np.testing.assert_array_equal(actual_ids, expected_ids)
        np.testing.assert_array_equal(actual_distances, expected_distances)


if __name__ == "__main__":
    unittest.main()
