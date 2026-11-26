# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import tempfile
import unittest

import numpy as np

from faiss.contrib.vecs_io import bvecs_iter, bvecs_iter_chunked


class TestBvecsIter(unittest.TestCase):

    @staticmethod
    def write_bvecs(path, vectors):
        n, d = vectors.shape
        records = np.empty((n, d + 4), dtype="uint8")
        records[:, :4] = np.array([d], dtype="<i4").view("uint8")
        records[:, 4:] = vectors
        records.tofile(path)

    def test_single_file(self):
        vectors = np.arange(35, dtype="uint8").reshape(7, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "vectors.bvecs")
            self.write_bvecs(path, vectors)
            batches = [batch.copy() for batch in bvecs_iter(path, batch_size=3)]
        self.assertEqual([len(batch) for batch in batches], [3, 3, 1])
        np.testing.assert_array_equal(np.vstack(batches), vectors)

    def test_numeric_chunk_order(self):
        with tempfile.TemporaryDirectory() as directory:
            # Reverse creation order also avoids depending on directory order.
            for number in reversed(range(12)):
                self.write_bvecs(
                    os.path.join(directory, f"chunk_{number}.bvecs"),
                    np.full((1, 3), number, dtype="uint8"),
                )
            batches = list(bvecs_iter_chunked(directory, batch_size=5))
        self.assertEqual([len(batch) for batch in batches], [5, 5, 2])
        np.testing.assert_array_equal(
            np.vstack(batches), np.repeat(np.arange(12)[:, None], 3, axis=1)
        )

    def test_batches_across_chunks(self):
        vectors = np.arange(84, dtype="uint8").reshape(28, 3)
        for sizes, batch_size in [([2, 3, 17, 6], 5), ([1] * 28, 10)]:
            with self.subTest(sizes=sizes, batch_size=batch_size):
                with tempfile.TemporaryDirectory() as directory:
                    start = 0
                    for number, size in enumerate(sizes):
                        self.write_bvecs(
                            os.path.join(directory, f"chunk_{number:04}.bvecs"),
                            vectors[start : start + size],
                        )
                        start += size
                    batches = [
                        batch.copy()
                        for batch in bvecs_iter_chunked(
                            directory, batch_size=batch_size
                        )
                    ]
                self.assertTrue(
                    all(len(batch) == batch_size for batch in batches[:-1])
                )
                np.testing.assert_array_equal(np.vstack(batches), vectors)

    def test_invalid_batch_size(self):
        # Validation must happen before file access or an infinite yield loop.
        for iterator in [bvecs_iter, bvecs_iter_chunked]:
            for batch_size in [0, -1]:
                with self.subTest(iterator=iterator, batch_size=batch_size):
                    with self.assertRaisesRegex(ValueError, "batch_size"):
                        next(iterator("unused", batch_size=batch_size))

    def test_batch_larger_than_database(self):
        with tempfile.TemporaryDirectory() as directory:
            vectors = np.array([[3, 2, 1]], dtype="uint8")
            self.write_bvecs(
                os.path.join(directory, "chunk_0000.bvecs"), vectors
            )
            batches = list(
                bvecs_iter_chunked(directory, batch_size=sys.maxsize)
            )
        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(batches[0], vectors)

    def test_missing_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            for number in [0, 2]:
                self.write_bvecs(
                    os.path.join(directory, f"chunk_{number}.bvecs"),
                    np.ones((1, 3), dtype="uint8"),
                )
            with self.assertRaisesRegex(ValueError, "Gap detected"):
                next(bvecs_iter_chunked(directory))

    def test_invalid_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_bvecs(
                os.path.join(directory, "chunk_0_extra.bvecs"),
                np.ones((1, 3), dtype="uint8"),
            )
            with self.assertRaisesRegex(ValueError, "Invalid chunk filename"):
                next(bvecs_iter_chunked(directory))


if __name__ == "__main__":
    unittest.main()
