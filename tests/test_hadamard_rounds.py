# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import struct
import unittest

import faiss
import numpy as np


def write_transform(transform):
    writer = faiss.VectorIOWriter()
    faiss.write_VectorTransform(transform, writer)
    return bytes(faiss.vector_to_array(writer.data))


def read_transform(data):
    reader = faiss.VectorIOReader()
    faiss.copy_array_to_vector(np.frombuffer(data, dtype="uint8"), reader.data)
    return faiss.read_VectorTransform(reader)


def hadamard_matrix(d):
    matrix = np.ones((1, 1))
    while len(matrix) < d:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(d)


class TestHadamardRounds(unittest.TestCase):
    def test_matrix_oracle(self):
        rng = np.random.RandomState(351)
        for d in (1, 3, 8, 17):
            x = rng.randn(7, d).astype("float32")
            for seed in (0, 42, 12345):
                for rounds in (1, 2, 3):
                    with self.subTest(d=d, seed=seed, rounds=rounds):
                        transform = faiss.HadamardRotation(d, seed, rounds)
                        matrix = hadamard_matrix(transform.d_out)
                        expected = np.pad(x, ((0, 0), (0, transform.d_out - d)))
                        for r in range(1, rounds + 1):
                            signs = faiss.vector_to_array(
                                getattr(transform, "signs%d" % r)
                            )
                            expected = (expected * signs) @ matrix
                        np.testing.assert_allclose(
                            transform.apply(x), expected, rtol=2e-5, atol=2e-6
                        )

    def test_default_legacy_result_and_nested_signs(self):
        # Frozen before adding the rounds argument, from upstream HRot seed 42.
        expected = np.array([
            [0.7071067094802856, 0.7071067690849304, -0.10101516544818878,
             0.7071067690849304, -0.30304574966430664, -0.7071067690849304,
             0.7071067094802856, -0.5050761699676514],
            [-1.717259168624878, -0.9091372489929199, -0.10101529210805893,
             -0.10101529210805893, -0.30304569005966187, 0.10101529210805893,
             -0.10101529210805893, -0.5050762891769409],
            [-4.14162540435791, -2.525381565093994, -0.10101545602083206,
             -0.9091372489929199, -0.3030456006526947, 0.909137487411499,
             -0.9091372489929199, -0.5050762891769409],
        ], dtype="float32")
        x = np.arange(24, dtype="float32").reshape(3, 8) / 7 - 1
        default = faiss.HadamardRotation(8, 42)
        self.assertEqual(default.rounds, 3)
        np.testing.assert_array_equal(default.apply(x), expected)
        for rounds in (1, 2, 3):
            transform = faiss.HadamardRotation(8, 42, rounds)
            for r in range(1, rounds + 1):
                np.testing.assert_array_equal(
                    faiss.vector_to_array(getattr(transform, "signs%d" % r)),
                    faiss.vector_to_array(getattr(default, "signs%d" % r)),
                )
            if rounds == 3:
                np.testing.assert_array_equal(transform.apply(x), expected)

    def test_norm_inner_product_and_inverse(self):
        rng = np.random.RandomState(53)
        for d in (1, 7, 16, 31):
            x = rng.randn(20, d).astype("float32")
            for rounds in (1, 2, 3):
                with self.subTest(d=d, rounds=rounds):
                    transform = faiss.HadamardRotation(d, 987, rounds)
                    y = transform.apply(x)
                    np.testing.assert_allclose(
                        y @ y.T, x @ x.T, rtol=2e-5, atol=2e-5
                    )
                    if transform.d_in == transform.d_out:
                        np.testing.assert_allclose(
                            transform.reverse_transform(y), x,
                            rtol=2e-5, atol=2e-6,
                        )
                    else:
                        with self.assertRaisesRegex(RuntimeError, "equal input/output"):
                            transform.reverse_transform(y)

    def test_empty_zero_batch_thread_and_object_consistency(self):
        old_threads = faiss.omp_get_max_threads()
        x = np.random.RandomState(12).randn(31, 13).astype("float32")
        try:
            for rounds in (1, 2, 3):
                transform = faiss.HadamardRotation(13, 42, rounds)
                self.assertEqual(transform.apply(x[:0]).shape, (0, 16))
                np.testing.assert_array_equal(
                    transform.apply(np.zeros_like(x)), np.zeros((31, 16))
                )
                faiss.omp_set_num_threads(1)
                expected = transform.apply(x)
                for threads in (1, 2, 4):
                    faiss.omp_set_num_threads(threads)
                    np.testing.assert_array_equal(transform.apply(x), expected)
                np.testing.assert_array_equal(
                    np.concatenate([transform.apply(x[:8]), transform.apply(x[8:])]),
                    expected,
                )
                np.testing.assert_array_equal(
                    faiss.HadamardRotation(13, 42, rounds).apply(x), expected
                )
        finally:
            faiss.omp_set_num_threads(old_threads)

    def test_invalid_constructor_and_mutable_rounds(self):
        x = np.ones((2, 8), dtype="float32")
        for rounds in (-1, 0, 4, 99):
            with self.subTest(rounds=rounds):
                with self.assertRaisesRegex(RuntimeError, "rounds"):
                    faiss.HadamardRotation(8, 42, rounds)
                transform = faiss.HadamardRotation(8)
                other = faiss.HadamardRotation(8)
                transform.rounds = rounds
                for operation in (
                    lambda: transform.init(42),
                    lambda: transform.train(x),
                    lambda: transform.apply(x),
                    lambda: transform.reverse_transform(x),
                    lambda: transform.check_identical(other),
                    lambda: other.check_identical(transform),
                    lambda: write_transform(transform),
                ):
                    with self.assertRaisesRegex(RuntimeError, "rounds"):
                        operation()

    def test_check_identical_includes_rounds(self):
        for rounds in (1, 2, 3):
            transform = faiss.HadamardRotation(8, 42, rounds)
            transform.check_identical(faiss.HadamardRotation(8, 42, rounds))
            with self.assertRaisesRegex(RuntimeError, "rounds"):
                transform.check_identical(faiss.HadamardRotation(8, 42, rounds % 3 + 1))

    def test_factory(self):
        for token, seed, rounds in (
            ("HR", 12345, 3), ("HR42", 42, 3),
            ("HRr1", 12345, 1), ("HRr2", 12345, 2),
            ("HRr3", 12345, 3), ("HR42r2", 42, 2),
            ("HR0r1", 0, 1),
        ):
            with self.subTest(token=token):
                index = faiss.index_factory(13, token + ",Flat")
                transform = faiss.downcast_VectorTransform(index.chain.at(0))
                self.assertEqual((transform.seed, transform.rounds), (seed, rounds))
                self.assertEqual(index.index.d, 16)
        for token in ("HRr0", "HRr4", "HRr-1", "HRr", "HR42r2x", "HRr2r1"):
            with self.subTest(token=token):
                with self.assertRaises(RuntimeError):
                    faiss.index_factory(13, token + ",Flat")

    def test_standalone_io_and_legacy_fixture(self):
        # Existing HRot payload: seed=42, d_in=d_out=8, is_trained=true.
        legacy = bytes.fromhex("48526f742a000000080000000800000001")
        transform = read_transform(legacy)
        self.assertEqual(transform.rounds, 3)
        self.assertEqual(write_transform(transform), legacy)
        self.assertEqual(write_transform(faiss.HadamardRotation(8, 42)), legacy)
        x = np.random.RandomState(54).randn(10, 13).astype("float32")
        for rounds in (1, 2, 3):
            transform = faiss.HadamardRotation(13, 42, rounds)
            data = write_transform(transform)
            self.assertEqual(data[:4], b"HRot" if rounds == 3 else b"HRnd")
            restored = read_transform(data)
            self.assertEqual(restored.rounds, rounds)
            transform.check_identical(restored)
            np.testing.assert_array_equal(transform.apply(x), restored.apply(x))
            self.assertEqual(write_transform(restored), data)

    def test_counted_io_rejects_corrupt_payloads(self):
        # Corrupt counts are rejected even before a common transform header.
        for rounds in (-1, 0, 4, 123456):
            with self.subTest(rounds=rounds):
                with self.assertRaisesRegex(RuntimeError, "rounds"):
                    read_transform(struct.pack("<4sIi", b"HRnd", 42, rounds))
        data = write_transform(faiss.HadamardRotation(8, 42, 2))
        for size in range(len(data)):
            with self.subTest(truncated=size):
                with self.assertRaises(RuntimeError):
                    read_transform(data[:size])
        for d_in, d_out in ((0, 0), (4, 3), (3, 8), (-1, 8), (8, -1)):
            with self.subTest(d_in=d_in, d_out=d_out):
                with self.assertRaises(RuntimeError):
                    read_transform(struct.pack("<4sIiii?", b"HRnd", 42, 2, d_in, d_out, True))

    def test_counted_io_preserves_allocation_limit(self):
        data = write_transform(faiss.HadamardRotation(1, 42, 1))
        old_limit = faiss.get_deserialization_vector_byte_limit()
        try:
            faiss.set_deserialization_vector_byte_limit(8)
            with self.assertRaisesRegex(RuntimeError, "byte limit"):
                read_transform(data)
        finally:
            faiss.set_deserialization_vector_byte_limit(old_limit)

    def test_flat_index_io_and_exact_search(self):
        rng = np.random.RandomState(68)
        for d in (8, 13):
            xb = rng.randn(100, d).astype("float32")
            xq = rng.randn(9, d).astype("float32")
            for metric in (faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT):
                exact = faiss.index_factory(d, "Flat", metric)
                exact.add(xb)
                expected_d, expected_i = exact.search(xq, 8)
                for rounds in (1, 2, 3):
                    with self.subTest(d=d, metric=metric, rounds=rounds):
                        index = faiss.index_factory(d, "HR42r%d,Flat" % rounds, metric)
                        index.add(xb)
                        distances, labels = index.search(xq, 8)
                        np.testing.assert_array_equal(labels, expected_i)
                        np.testing.assert_allclose(distances, expected_d, rtol=1e-5, atol=1e-5)
                        restored = faiss.deserialize_index(faiss.serialize_index(index))
                        restored_d, restored_i = restored.search(xq, 8)
                        np.testing.assert_array_equal(restored_i, labels)
                        np.testing.assert_array_equal(restored_d, distances)
                        if d == 8:
                            np.testing.assert_allclose(restored.reconstruct(3), xb[3], atol=2e-6)


if __name__ == "__main__":
    unittest.main()
