# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Caller-owned label buffers in the public NumPy e-means streaming API."""

import unittest

import faiss
import numpy as np

from faiss.contrib.e_means import NumpyEmeans


def _sample():
    rng = np.random.default_rng(20261111)
    left = rng.normal([0., 0., 0.], .12, size=(50, 3))
    right = rng.normal([10., 0., 0.], .12, size=(50, 3))
    x = np.vstack([left, right]).astype("float32")
    initial = np.array([[0., 0., 0.], [10., 0., 0.],
                        [100., 100., 100.]], dtype="float32")
    return x[rng.permutation(len(x))], initial


def _model(threshold, capture=False):
    return NumpyEmeans(
        3, 3, nepochs=2, batch_size=20, alpha0=.2, seed=1234,
        unused_threshold=threshold, capture_assignments=capture,
        assignment_chunk_size=5,
    )


class TestCallerOwnedLabels(unittest.TestCase):

    def test_labels_out_preserves_revival_trajectories(self):
        x, initial = _sample()
        for threshold in (0., .1):
            with self.subTest(threshold=threshold):
                ordinary = _model(threshold)
                observed = _model(threshold)
                ordinary.begin(len(x), initial)
                observed.begin(len(x), initial)
                n_assignment_calls = [0, 0]
                for j, model in enumerate((ordinary, observed)):
                    original = model._assign_rows

                    def counted(*args, _original=original, _j=j):
                        n_assignment_calls[_j] += 1
                        return _original(*args)

                    model._assign_rows = counted

                for epoch in range(2):
                    for start in range(0, len(x), 20):
                        batch = x[start:start + 20]
                        frame = observed.current_centroids().copy()
                        old_stats = new_stats = None
                        for chunk in (batch[:3], batch[3:10], batch[10:]):
                            old_stats = ordinary.accumulate(chunk)
                            labels = np.full(len(chunk), -1, dtype="int64")
                            new_stats = observed.accumulate(
                                chunk, labels_out=labels)
                            self.assertTrue(np.shares_memory(
                                new_stats["labels"], labels))
                            _, reference = faiss.knn(chunk, frame, 1)
                            np.testing.assert_array_equal(
                                labels, reference.ravel())
                        self.assertEqual(n_assignment_calls[0],
                                         n_assignment_calls[1])
                        self.assertEqual(new_stats["n"], 20)
                        np.testing.assert_array_equal(
                            old_stats["sums"], new_stats["sums"])
                        np.testing.assert_array_equal(
                            old_stats["counts"], new_stats["counts"])
                        self.assertEqual(old_stats["objective"],
                                         new_stats["objective"])
                        ordinary.update(old_stats)
                        observed.update(new_stats)
                        np.testing.assert_array_equal(
                            ordinary.current_centroids(),
                            observed.current_centroids())
                        np.testing.assert_array_equal(
                            ordinary._usage, observed._usage)
                        self.assertEqual(
                            ordinary.n_splits, observed.n_splits)
                        self.assertEqual(
                            ordinary._objective, observed._objective)
                    ordinary.end_epoch()
                    observed.end_epoch()
                    self.assertEqual(
                        ordinary.iteration_stats[-1]["obj"],
                        observed.iteration_stats[-1]["obj"])
                np.testing.assert_array_equal(
                    ordinary.finalize(), observed.finalize())
                np.testing.assert_array_equal(ordinary.obj, observed.obj)
                if threshold:
                    self.assertGreater(observed.n_splits, 0)
                else:
                    self.assertEqual(observed.n_splits, 0)

    def test_partial_batch_labels_are_caller_owned(self):
        x, initial = _sample()
        model = _model(0.)
        model.begin(len(x), initial)
        out = np.empty(3, dtype=np.int64)
        partial = model.accumulate(x[:3], labels_out=out)
        self.assertEqual(partial["n"], 3)
        self.assertIsNone(partial["counts"])
        self.assertTrue(np.shares_memory(out, partial["labels"]))
        _, expected = faiss.knn(x[:3], initial, 1)
        np.testing.assert_array_equal(out, expected.ravel())

    def test_empty_chunk_preserves_caller_owned_label_view(self):
        x, initial = _sample()
        model = _model(0.)
        model.begin(len(x), initial)
        first = np.empty(3, dtype=np.int64)
        model.accumulate(x[:3], labels_out=first)
        empty = np.empty(0, dtype=np.int64)
        stats = model.accumulate(x[:0], labels_out=empty)
        self.assertEqual(stats["n"], 3)
        self.assertIsNone(stats["counts"])
        self.assertEqual(stats["labels"].shape, (0,))
        self.assertIs(stats["labels"].base, empty)
        without_output = model.accumulate(x[:0])
        self.assertEqual(without_output["n"], 3)
        self.assertNotIn("labels", without_output)

    def test_capture_enabled_prefers_supplied_buffer_on_final_epoch(self):
        x = np.array([[0.], [2.], [10.], [12.]], dtype=np.float32)
        model = NumpyEmeans(1, 2, nepochs=1, batch_size=4,
                            alpha0=.5, capture_assignments=True)
        model.begin(len(x), np.array([[0.], [10.]], dtype=np.float32))
        out = np.full(len(x), -1, dtype=np.int64)
        stats = model.accumulate(x, labels_out=out)
        np.testing.assert_array_equal(out, [0, 0, 1, 1])
        self.assertTrue(np.shares_memory(stats["labels"], out))
        model.update(stats)
        model.end_epoch()
        np.testing.assert_array_equal(model.finalize(), [[1.], [11.]])

    def test_empty_chunk_during_final_epoch_capture(self):
        x = np.array([[0.], [2.], [10.], [12.]], dtype=np.float32)
        model = NumpyEmeans(
            1, 2, nepochs=1, batch_size=4, alpha0=.5,
            capture_assignments=True,
        )
        model.begin(len(x), np.array([[0.], [10.]], dtype=np.float32))
        partial = model.accumulate(x[:2])
        np.testing.assert_array_equal(partial["labels"], [0, 0])
        default_empty = model.accumulate(x[:0])
        self.assertEqual(default_empty["labels"].shape, (0,))
        supplied = np.empty(0, dtype=np.int64)
        explicit_empty = model.accumulate(x[:0], labels_out=supplied)
        self.assertIs(explicit_empty["labels"].base, supplied)
        completed = model.accumulate(x[2:])
        np.testing.assert_array_equal(completed["labels"], [1, 1])
        model.update(completed)
        model.end_epoch()
        np.testing.assert_array_equal(model.finalize(), [[1.], [11.]])

    def test_spherical_numpy_labels_use_inner_product_assignments(self):
        x = np.array([[2., .1], [1., .5], [.1, 2.],
                      [-1., .5]], dtype=np.float32)
        initial = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        model = NumpyEmeans(2, 2, nepochs=1, batch_size=4, alpha0=.5,
                            spherical=True, unused_threshold=0.,
                            capture_assignments=False)
        model.begin(len(x), initial)
        out = np.empty(len(x), dtype=np.int64)
        _, expected = faiss.knn(
            x, model.current_centroids(), 1,
            metric=faiss.METRIC_INNER_PRODUCT)
        stats = model.accumulate(x, labels_out=out)
        np.testing.assert_array_equal(out, expected.ravel())
        self.assertTrue(np.shares_memory(stats["labels"], out))
        model.update(stats)
        model.end_epoch()
        self.assertEqual(model.finalize().shape, (2, 2))

    def test_disjoint_views_of_one_allocation_are_allowed(self):
        # Input has two strided float32 rows, at byte offsets 0 and 24.
        # The int64 output occupies bytes [4, 20), so it is disjoint.
        # Conservative may_share_memory() reports a false positive.
        allocation = np.arange(8, dtype=np.float32)
        x = allocation[::6, None]
        labels = np.ndarray((2,), dtype=np.int64,
                            buffer=allocation.data, offset=4)
        self.assertTrue(np.may_share_memory(x, labels))
        self.assertFalse(np.shares_memory(x, labels))
        self.assertTrue(labels.flags.c_contiguous)
        model = NumpyEmeans(1, 2, nepochs=1, batch_size=2,
                            alpha0=.5, unused_threshold=0.)
        model.begin(2, np.array([[0.], [6.]], dtype=np.float32))
        stats = model.accumulate(x, labels_out=labels)
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_array_equal(x.ravel(), [0., 6.])
        self.assertEqual(stats["n"], 2)

    def test_rejects_invalid_buffers_without_poisoning_model(self):
        x = np.array([[0., 0.], [1., 0.],
                      [10., 0.], [11., 0.]], dtype=np.float32)
        model = NumpyEmeans(2, 2, nepochs=1, batch_size=4, alpha0=.5)
        model.begin(len(x), np.array([[0., 0.], [10., 0.]],
                                      dtype=np.float32))
        readonly = np.empty(len(x), dtype=np.int64)
        readonly.flags.writeable = False
        invalid = [
            [0] * len(x),
            np.empty(len(x), dtype=np.int32),
            np.empty(len(x) - 1, dtype=np.int64),
            np.empty(2 * len(x), dtype=np.int64)[::2],
            readonly,
            x.view(np.int64).reshape(-1),
        ]
        for buffer in invalid:
            with self.subTest(kind=type(buffer).__name__):
                with self.assertRaises(ValueError):
                    model.accumulate(x, labels_out=buffer)
                self.assertIsNone(model._pending_stats)
                self.assertFalse(model._failed)
        out = np.empty(len(x), dtype=np.int64)
        stats = model.accumulate(x, labels_out=out)
        self.assertEqual(stats["n"], len(x))
        np.testing.assert_array_equal(out, [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
