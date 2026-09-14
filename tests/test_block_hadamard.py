# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import unittest

import faiss
import numpy as np


def hadamard_matrix(d):
    matrix = np.ones((1, 1), dtype="float32")
    while len(matrix) < d:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(np.float32(d))


def block_hadamard_matrix(d):
    matrix = np.zeros((d, d), dtype="float32")
    offset = 0
    remaining = d
    while remaining:
        block = 1 << (remaining.bit_length() - 1)
        matrix[offset : offset + block, offset : offset + block] = (
            hadamard_matrix(block)
        )
        offset += block
        remaining -= block
    return matrix


class TestBlockHadamardRotation(unittest.TestCase):
    def test_small_explicit_matrix_oracle(self):
        rs = np.random.RandomState(10)
        d = 6  # 4 + 2
        transform = faiss.BlockHadamardRotation(d, 42)
        permutation = faiss.vector_to_array(transform.permutation)
        signs = faiss.vector_to_array(transform.signs)
        x = rs.randn(5, d).astype("float32")

        permuted = x[:, permutation] * signs
        expected = permuted @ block_hadamard_matrix(d).T
        np.testing.assert_allclose(
            transform.apply(x), expected, rtol=2e-6, atol=2e-6
        )

    def test_preserves_norm_inner_product_and_l2(self):
        rs = np.random.RandomState(11)
        for d in (1, 3, 6, 7, 128, 768, 960, 1536):
            x = rs.randn(4, d).astype("float32")
            y = faiss.BlockHadamardRotation(d, 100 + d).apply(x)
            np.testing.assert_allclose(
                np.linalg.norm(y, axis=1),
                np.linalg.norm(x, axis=1),
                rtol=3e-5,
                atol=3e-5,
            )
            np.testing.assert_allclose(
                y @ y.T, x @ x.T, rtol=5e-5, atol=5e-4
            )
            np.testing.assert_allclose(
                ((y[:, None] - y[None, :]) ** 2).sum(axis=2),
                ((x[:, None] - x[None, :]) ** 2).sum(axis=2),
                rtol=5e-5,
                atol=5e-4,
            )

    def test_inverse_and_in_place_apply(self):
        rs = np.random.RandomState(12)
        for d in (1, 3, 6, 7, 128, 768):
            x = rs.randn(3, d).astype("float32")
            transform = faiss.BlockHadamardRotation(d, 200 + d)
            transformed = transform.apply(x)
            reconstructed = transform.reverse_transform(transformed)
            np.testing.assert_allclose(
                reconstructed, x, rtol=3e-5, atol=3e-5
            )

            in_place = x.copy()
            transform.apply_noalloc(
                len(in_place),
                faiss.swig_ptr(in_place),
                faiss.swig_ptr(in_place),
            )
            np.testing.assert_array_equal(in_place, transformed)

    def test_empty_single_batch_and_concurrent_apply(self):
        rs = np.random.RandomState(13)
        d = 63
        transform = faiss.BlockHadamardRotation(d, 99)
        empty = np.empty((0, d), dtype="float32")
        self.assertEqual(transform.apply(empty).shape, (0, d))

        x = rs.randn(32, d).astype("float32")
        expected = transform.apply(x)
        np.testing.assert_array_equal(transform.apply(x[:1]), expected[:1])
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            chunks = list(executor.map(transform.apply, np.split(x, 4)))
        np.testing.assert_array_equal(np.vstack(chunks), expected)

    def test_seed_and_saved_metadata(self):
        first = faiss.BlockHadamardRotation(31, 123)
        second = faiss.BlockHadamardRotation(31, 123)
        third = faiss.BlockHadamardRotation(31, 124)
        first.check_identical(second)
        with self.assertRaisesRegex(RuntimeError, "seeds must match"):
            first.check_identical(third)
        np.testing.assert_array_equal(
            faiss.vector_to_array(first.permutation),
            faiss.vector_to_array(second.permutation),
        )
        np.testing.assert_array_equal(
            faiss.vector_to_array(first.signs),
            faiss.vector_to_array(second.signs),
        )
        self.assertFalse(
            np.array_equal(
                faiss.vector_to_array(first.permutation),
                faiss.vector_to_array(third.permutation),
            )
            and np.array_equal(
                faiss.vector_to_array(first.signs),
                faiss.vector_to_array(third.signs),
            )
        )

    def test_clone_and_io_preserve_actual_arrays(self):
        rs = np.random.RandomState(14)
        x = rs.randn(5, 33).astype("float32")
        transform = faiss.BlockHadamardRotation(33, 55)
        reference = transform.apply(x)
        # Arrays, rather than seed regeneration, are authoritative.
        transform.seed = 999

        storage = faiss.IndexFlatL2(33)
        index = faiss.IndexPreTransform(transform, storage)
        index.own_fields = False
        cloned_index = faiss.clone_index(index)
        cloned = faiss.downcast_VectorTransform(cloned_index.chain.at(0))
        np.testing.assert_array_equal(cloned.apply(x), reference)
        self.assertEqual(cloned.seed, 999)

        writer = faiss.VectorIOWriter()
        faiss.write_VectorTransform(transform, writer)
        reader = faiss.VectorIOReader()
        faiss.copy_array_to_vector(
            faiss.vector_to_array(writer.data), reader.data
        )
        restored = faiss.read_VectorTransform(reader)
        self.assertIsInstance(restored, faiss.BlockHadamardRotation)
        self.assertEqual(restored.seed, 999)
        np.testing.assert_array_equal(restored.apply(x), reference)
        np.testing.assert_array_equal(
            faiss.vector_to_array(restored.permutation),
            faiss.vector_to_array(transform.permutation),
        )
        np.testing.assert_array_equal(
            faiss.vector_to_array(restored.signs),
            faiss.vector_to_array(transform.signs),
        )

    def test_io_budget_uses_linear_metadata_size(self):
        transform = faiss.BlockHadamardRotation(768, 42)
        writer = faiss.VectorIOWriter()
        faiss.write_VectorTransform(transform, writer)
        serialized = faiss.vector_to_array(writer.data)
        byte_limit = 1 << 20
        self.assertLess(serialized.nbytes, byte_limit)

        old_limit = faiss.get_deserialization_vector_byte_limit()
        try:
            faiss.set_deserialization_vector_byte_limit(byte_limit)
            reader = faiss.VectorIOReader()
            faiss.copy_array_to_vector(serialized, reader.data)
            restored = faiss.read_VectorTransform(reader)
        finally:
            faiss.set_deserialization_vector_byte_limit(old_limit)

        self.assertIsInstance(restored, faiss.BlockHadamardRotation)
        np.testing.assert_array_equal(
            faiss.vector_to_array(restored.permutation),
            faiss.vector_to_array(transform.permutation),
        )
        np.testing.assert_array_equal(
            faiss.vector_to_array(restored.signs),
            faiss.vector_to_array(transform.signs),
        )

    def test_normalization_prevents_intermediate_overflow(self):
        d = 128
        transform = faiss.BlockHadamardRotation(d, 42)
        permutation = faiss.vector_to_array(transform.permutation)
        signs = faiss.vector_to_array(transform.signs)
        x = np.empty((1, d), dtype="float32")
        x[0, permutation] = signs * np.float32(1e37)

        actual = transform.apply(x)
        expected = np.zeros((1, d), dtype="float32")
        expected[0, 0] = np.float32(np.sqrt(np.float32(d)) * np.float32(1e37))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)

    def test_rejects_invalid_dimensions_and_metadata(self):
        for d in (0, -1):
            with self.assertRaisesRegex(RuntimeError, "positive dimensions"):
                faiss.BlockHadamardRotation(d)

        x = np.ones((1, 8), dtype="float32")
        output = np.full_like(x, 12345.0)
        transform = faiss.BlockHadamardRotation(8, 1)
        faiss.copy_array_to_vector(
            np.arange(7, dtype="int32"), transform.permutation
        )
        with self.assertRaisesRegex(RuntimeError, "size must match"):
            transform.apply_noalloc(
                1, faiss.swig_ptr(x), faiss.swig_ptr(output)
            )
        np.testing.assert_array_equal(output, np.full_like(x, 12345.0))

        transform = faiss.BlockHadamardRotation(8, 1)
        faiss.copy_array_to_vector(
            np.array([0, 0, 2, 3, 4, 5, 6, 7], dtype="int32"),
            transform.permutation,
        )
        writer = faiss.VectorIOWriter()
        faiss.write_VectorTransform(transform, writer)
        reader = faiss.VectorIOReader()
        faiss.copy_array_to_vector(
            faiss.vector_to_array(writer.data), reader.data
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            faiss.read_VectorTransform(reader)

        transform = faiss.BlockHadamardRotation(8, 1)
        signs = faiss.vector_to_array(transform.signs)
        signs[3] = 0.0
        faiss.copy_array_to_vector(signs, transform.signs)
        writer = faiss.VectorIOWriter()
        faiss.write_VectorTransform(transform, writer)
        reader = faiss.VectorIOReader()
        faiss.copy_array_to_vector(
            faiss.vector_to_array(writer.data), reader.data
        )
        with self.assertRaisesRegex(RuntimeError, "invalid.*sign"):
            faiss.read_VectorTransform(reader)

        transform = faiss.BlockHadamardRotation(8, 1)
        transform.d_out = 7
        writer = faiss.VectorIOWriter()
        faiss.write_VectorTransform(transform, writer)
        reader = faiss.VectorIOReader()
        faiss.copy_array_to_vector(
            faiss.vector_to_array(writer.data), reader.data
        )
        with self.assertRaisesRegex(RuntimeError, "invalid.*dimensions"):
            faiss.read_VectorTransform(reader)

    def test_factory_index_search_and_filter(self):
        rs = np.random.RandomState(15)
        d = 31
        xb = rs.randn(200, d).astype("float32")
        xq = rs.randn(5, d).astype("float32")
        index = faiss.index_factory(d, "BHR42,Flat")
        index.train(xb)
        index.add(xb)

        transform = faiss.downcast_VectorTransform(index.chain.at(0))
        self.assertIsInstance(transform, faiss.BlockHadamardRotation)
        inner = faiss.downcast_index(index.index)
        allowed = np.array(
            [1, 7, 19, 23, 41, 88, 101, 150, 177, 199], dtype="int64"
        )
        selector = faiss.IDSelectorBatch(allowed)
        inner_params = faiss.SearchParameters()
        inner_params.sel = selector
        outer_params = faiss.SearchParametersPreTransform()
        outer_params.index_params = inner_params

        expected_distances, expected_ids = inner.search(
            transform.apply(xq), 5, params=inner_params
        )
        actual_distances, actual_ids = index.search(xq, 5, params=outer_params)
        np.testing.assert_array_equal(actual_ids, expected_ids)
        np.testing.assert_array_equal(actual_distances, expected_distances)
        self.assertTrue(np.isin(actual_ids, allowed).all())

        restored = faiss.deserialize_index(faiss.serialize_index(index))
        restored_distances, restored_ids = restored.search(
            xq, 5, params=outer_params
        )
        np.testing.assert_array_equal(restored_ids, actual_ids)
        np.testing.assert_array_equal(restored_distances, actual_distances)


if __name__ == "__main__":
    unittest.main()
