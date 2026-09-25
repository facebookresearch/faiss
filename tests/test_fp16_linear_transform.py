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

    def test_invalid_bias_fails_before_writing_output(self):
        self.require_fp16()
        d = 8
        transform = faiss.LinearTransform(d, d, True)
        faiss.copy_array_to_vector(np.eye(d, dtype="float32").ravel(), transform.A)
        transform.is_trained = True
        transform.prepare_fp16()
        query = np.ones(d, dtype="float32")

        for bias in (
            np.empty(0, dtype="float32"),
            np.arange(d - 1, dtype="float32"),
        ):
            faiss.copy_array_to_vector(bias, transform.b)
            output = np.full(d, np.float32(12345.0))
            with self.assertRaisesRegex(RuntimeError, "Bias not initialized"):
                transform.apply_noalloc_fp16(
                    1, faiss.swig_ptr(query), faiss.swig_ptr(output)
                )
            np.testing.assert_array_equal(output, np.float32(12345.0))

        bias = np.arange(d, dtype="float32")
        faiss.copy_array_to_vector(bias, transform.b)
        output = np.empty(d, dtype="float32")
        transform.apply_noalloc_fp16(
            1, faiss.swig_ptr(query), faiss.swig_ptr(output)
        )
        np.testing.assert_array_equal(output, query + bias)

    def test_matrix_range_and_failed_cache_refresh(self):
        self.require_fp16()
        d = 8
        transform = faiss.LinearTransform(d, d)
        matrix = np.eye(d, dtype="float32")
        faiss.copy_array_to_vector(matrix.ravel(), transform.A)
        transform.is_trained = True
        transform.prepare_fp16()

        query = np.ones(d, dtype="float32")
        output = np.empty(d, dtype="float32")
        for invalid in (100000.0, -100000.0, np.inf, -np.inf, np.nan):
            broken = matrix.copy()
            broken[0, 0] = invalid
            faiss.copy_array_to_vector(broken.ravel(), transform.A)
            with self.assertRaisesRegex(
                RuntimeError, "non-finite or out-of-range"
            ):
                transform.prepare_fp16()
            with self.assertRaisesRegex(RuntimeError, "cache is missing or stale"):
                transform.apply_noalloc_fp16(
                    1, faiss.swig_ptr(query), faiss.swig_ptr(output)
                )

        boundary = np.zeros((d, d), dtype="float32")
        boundary[0, 0] = 65504.0
        boundary[1, 1] = -65504.0
        faiss.copy_array_to_vector(boundary.ravel(), transform.A)
        transform.prepare_fp16()
        transform.apply_noalloc_fp16(
            1, faiss.swig_ptr(query), faiss.swig_ptr(output)
        )
        np.testing.assert_array_equal(output, boundary @ query)

    def test_query_range_falls_back_per_row(self):
        self.require_fp16()
        d = 7
        transform = faiss.LinearTransform(d, d)
        matrix = np.eye(d, dtype="float32")
        faiss.copy_array_to_vector(matrix.ravel(), transform.A)
        transform.is_trained = True
        transform.prepare_fp16()

        queries = np.array(
            [
                np.linspace(-1.0, 1.0, d),
                np.full(d, 100000.0),
                np.full(d, -100000.0),
                np.full(d, 65504.0),
                np.full(d, -65504.0),
                np.zeros(d),
            ],
            dtype="float32",
        )
        expected = np.empty_like(queries)
        actual = np.empty_like(queries)
        transform.apply_noalloc(
            len(queries), faiss.swig_ptr(queries), faiss.swig_ptr(expected)
        )
        transform.apply_noalloc_fp16(
            len(queries), faiss.swig_ptr(queries), faiss.swig_ptr(actual)
        )
        np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)

        non_finite = np.zeros((4, d), dtype="float32")
        non_finite[0, 0] = np.inf
        non_finite[1, 0] = -np.inf
        non_finite[2, 0] = np.nan
        non_finite[3, 0] = -0.0
        transform.apply_noalloc(
            len(non_finite),
            faiss.swig_ptr(non_finite),
            faiss.swig_ptr(expected[: len(non_finite)]),
        )
        transform.apply_noalloc_fp16(
            len(non_finite),
            faiss.swig_ptr(non_finite),
            faiss.swig_ptr(actual[: len(non_finite)]),
        )
        np.testing.assert_allclose(
            actual[: len(non_finite)],
            expected[: len(non_finite)],
            rtol=5e-4,
            atol=5e-4,
            equal_nan=True,
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

    def test_single_large_query_search_falls_back_to_fp32(self):
        self.require_fp16()
        rs = np.random.RandomState(303)
        d = 17
        xb = rs.randn(200, d).astype("float32")
        xq = np.full((1, d), 100000.0, dtype="float32")
        transform = faiss.LinearTransform(d, d)
        faiss.copy_array_to_vector(
            np.eye(d, dtype="float32").ravel(), transform.A
        )
        transform.is_trained = True
        transform.prepare_fp16()
        storage = faiss.IndexFlatL2(d)
        index = faiss.IndexPreTransform(transform, storage)
        index.own_fields = False
        index.add(xb)

        expected_distances, expected_ids = index.search(xq, 10)
        params = faiss.SearchParametersPreTransform()
        params.use_fp16_transform = True
        actual_distances, actual_ids = index.search(xq, 10, params=params)
        np.testing.assert_array_equal(actual_ids, expected_ids)
        np.testing.assert_array_equal(actual_distances, expected_distances)


if __name__ == "__main__":
    unittest.main()
