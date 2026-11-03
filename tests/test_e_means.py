# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import subprocess
import sys
import faiss.contrib.e_means

import numpy as np

from faiss.contrib.e_means import Emeans
from faiss.contrib.e_means.e_means import (
    _epoch_rows,
    _epoch_seed,
    _resolve_recipe,
    _scheduled_alpha,
)
from e_means_test_utils import EmeansReferenceContracts


class TestEmeansRecipe(unittest.TestCase):
    def test_fractional_epoch_rounding_and_validation(self):
        self.assertEqual(_epoch_rows(10_000_000_000, 1.9999999995),
                         [9_999_999_995, 10_000_000_000])
        self.assertEqual(_epoch_rows(3, 1.5), [2, 3])
        self.assertEqual(_epoch_rows(5, 1.5), [2, 5])
        self.assertEqual(_epoch_rows(3, 1.01), [3])
        for value in (0, -1, float('nan'), float('inf'), -float('inf')):
            with self.subTest(niter=value), self.assertRaises(ValueError):
                Emeans(1, 1, niter=value)
        with self.assertRaisesRegex(ValueError, 'rounds to zero'):
            Emeans(1, 1, niter=0.1).begin(3, np.zeros((1, 1), dtype='float32'))

    def test_reference_values(self):
        cases = [
            (1_000_000, 16_384, 25, "throughput", 262_144, 1.0),
            (1_000_000, 16_384, 25, "quality", 32_768, 0.1545388281304627),
            (100_000_000, 262_144, 25, "throughput", 33_554_432, 1.0),
            (
                100_000_000,
                262_144,
                25,
                "quality",
                4_194_304,
                0.42718555053293517,
            ),
        ]
        for case in cases:
            n, k, niter, operating_point, expected_batch, expected_alpha = case
            with self.subTest(n=n, k=k, operating_point=operating_point):
                batch, alpha = _resolve_recipe(
                    n, k, niter, operating_point, None, None
                )
                self.assertEqual(batch, expected_batch)
                self.assertAlmostEqual(alpha, expected_alpha)

    def test_invalid_overrides(self):
        with self.assertRaises(ValueError):
            Emeans(4, 3, batch_size=16)
        with self.assertRaises(ValueError):
            Emeans(4, 3, alpha0=0.2)
        with self.assertRaises(ValueError):
            Emeans(4, 3, batch_size=0, alpha0=0.2)
        with self.assertRaises(ValueError):
            Emeans(4, 3, batch_size=16, alpha0=0.0)
        for name in ("assignment_chunk_size", "centroid_chunk_size"):
            for value in (0, -1):
                with self.subTest(parameter=name, value=value), self.assertRaises(ValueError):
                    Emeans(4, 3, **{name: value})

    def test_derived_recipe_metadata(self):
        x = np.arange(124, dtype="float32").reshape(31, 4)
        km = Emeans(4, 3, niter=3, assignment_chunk_size=7)

        km.train(x)

        self.assertEqual(km.recipe["name"], "canonical_v1")
        batch_size = km.recipe["batch_size"]
        steps_per_pass = (len(x) + batch_size - 1) // batch_size
        self.assertEqual(km.recipe["steps_per_pass"], steps_per_pass)
        self.assertEqual(km.recipe["total_steps"], 3 * steps_per_pass)
        self.assertEqual(km.recipe["tail_policy"], "keep")
        self.assertEqual(km.recipe["assignment_chunk_size"], 7)

    def test_schedule_uses_logical_steps(self):
        alpha0 = 0.4
        self.assertEqual(_scheduled_alpha(alpha0, 0, 10), alpha0)
        self.assertAlmostEqual(_scheduled_alpha(alpha0, 10, 10), 0.04)
        last = _scheduled_alpha(alpha0, 9, 10)
        self.assertGreater(last, 0.04)
        self.assertAlmostEqual(last, 0.04880982706687237)


class TestEmeans(EmeansReferenceContracts, unittest.TestCase):
    def reference_arrays(self):
        return [("numpy", lambda values, dtype="float32": np.asarray(values, dtype=dtype))]

    def test_numpy_training_does_not_import_torch(self):
        # A separate interpreter proves optional-dependency isolation even
        # when the surrounding test runner has already imported Torch.
        code = """
import builtins
import pathlib
import sys
import numpy as np
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('NumPy training imported Torch')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
from faiss.contrib.e_means import Emeans
import faiss.contrib.e_means
assert pathlib.Path(faiss.contrib.e_means.__file__).resolve() == pathlib.Path(sys.argv[1]).resolve()
model = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)
model.begin(4, np.array([[0], [10]], dtype='float32'))
np.testing.assert_array_equal(model.current_centroids(), [[0], [10]])
centroids, captured_labels = model.train(np.array([[0], [2], [10], [12]], dtype='float32'),
                        np.array([[0], [10]], dtype='float32'))
assert model.obj[-1] == 8
assert centroids is model.centroids and captured_labels is model.assignments
np.testing.assert_array_equal(model.centroids, [[1], [11]])
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code,
                        faiss.contrib.e_means.__file__], check=True, timeout=30)

    def test_large_batch_sums_and_closing_mean(self):
        n = 2**24 + 2
        x = np.full((n, 1), 0.25, dtype="float32")
        km = Emeans(1, 1, niter=1, batch_size=n, alpha0=1.0)
        km.begin(n, np.zeros((1, 1), dtype="float32"))
        partial = km.accumulate(x[:2**24])
        self.assertEqual(partial["n"], 2**24)
        self.assertIsNone(partial["sums"])
        self.assertIsNone(partial["counts"])
        stats = km.accumulate(x[2**24:])
        self.assertEqual(stats["sums"].dtype, np.dtype("float32"))
        self.assertEqual(stats["counts"].dtype, np.dtype("float32"))
        self.assertEqual(stats["sums"][0, 0], n * 0.25)
        self.assertEqual(stats["counts"][0], n)
        km.update(stats)
        km.end_epoch()
        self.assertEqual(km.finalize()[0, 0], 0.25)

    def test_closing_mean_and_assign(self):
        x = np.array([[0.0], [2.0], [10.0], [12.0]], dtype="float32")
        init = np.array([[0.0], [10.0]], dtype="float32")
        km = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)

        centroids, captured_labels = km.train(x, init_centroids=init)

        np.testing.assert_array_equal(km.centroids, [[1.0], [11.0]])
        self.assertEqual(km.obj[-1], 8.0)
        self.assertIs(centroids, km.centroids)
        self.assertIs(captured_labels, km.assignments)
        self.assertEqual(km.recipe["name"], "manual")
        self.assertIsNone(km.recipe["operating_point"])
        distance, labels = km.assign(x)
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])
        np.testing.assert_array_equal(distance, [1.0, 1.0, 1.0, 1.0])

    def test_train_tuple_capture_opt_out_and_owned_lifetime(self):
        x = np.array([[0.], [2.], [10.], [12.]], dtype="float32")
        initial = np.array([[0.], [10.]], dtype="float32")
        for capture in (None, False):
            options = {} if capture is None else dict(capture_assignments=capture)
            model = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5, **options)
            calls, original = [], model._assign_rows

            def assignment(*args, _calls=calls, _original=original, **kwargs):
                _calls.append(True)
                return _original(*args, **kwargs)

            model._assign_rows = assignment
            centroids, labels = model.train(x, initial)
            self.assertEqual(len(calls), 1)  # No extra closing-centroid search.
            self.assertIs(centroids, model.centroids)
            self.assertIs(labels, model.assignments)
            self.assertEqual(model.obj[-1], 8.)
            self.assertEqual(model.assignment_backend, "auto")
            self.assertEqual(model.recipe["assignment_backend"], "native")
            if capture is None:
                self.assertEqual(labels.dtype, np.dtype("int64"))
                np.testing.assert_array_equal(labels, [0, 0, 1, 1])
            else:
                self.assertIsNone(labels)
            saved_centroids = centroids.copy()
            saved_labels = None if labels is None else labels.copy()
            model.reset()
            model.train(x + 100, initial + 100)
            np.testing.assert_array_equal(centroids, saved_centroids)
            if labels is not None:
                np.testing.assert_array_equal(labels, saved_labels)

    def test_resident_capture_memory_is_bounded_by_assignment_chunks(self):
        # PR3's memory witness: a full-dataset logical batch must not allocate
        # another full-dataset label array before scattering its assignments.
        import tracemalloc

        n = 400_000
        x = np.random.RandomState(0).randn(n, 2).astype("float32")
        Emeans(2, 16, niter=1, batch_size=1000, alpha0=0.5).train(x[:1000])

        def peak(capture):
            tracemalloc.start()
            try:
                base = tracemalloc.get_traced_memory()[0]
                Emeans(2, 16, niter=1, batch_size=n, alpha0=0.5, seed=3,
                       capture_assignments=capture).train(x)
                return tracemalloc.get_traced_memory()[1] - base
            finally:
                tracemalloc.stop()

        extra_bytes_per_row = (peak(True) - peak(False)) / n
        self.assertTrue(6 < extra_bytes_per_row < 10, extra_bytes_per_row)

    def test_untouched_centroid_retains_its_position(self):
        x = np.arange(4, dtype="float32")[:, None]
        init = np.array([[0.0], [100.0]], dtype="float32")
        first = Emeans(1, 2, niter=1, batch_size=4, alpha0=1.0)
        first.train(x, init_centroids=init)
        np.testing.assert_array_equal(first.centroids, [[1.5], [100.0]])

        second = Emeans(1, 2, niter=2, batch_size=4, alpha0=1.0)
        second.train(x, init_centroids=init)
        # A second pass must retain the never-assigned centroid's position.
        np.testing.assert_array_equal(second.centroids, [[1.5], [100.0]])

    def test_caller_batches_match_matrix_training(self):
        x = np.random.RandomState(123).randn(17, 3).astype("float32")
        init = x[:4].copy()
        args = dict(niter=3, batch_size=4, alpha0=0.2, seed=7,
                    assignment_chunk_size=2)
        matrix, caller = Emeans(3, 4, **args), Emeans(3, 4, **args)
        matrix.train(x, init_centroids=init)
        caller.begin(len(x), init)
        for epoch in range(caller.niter):
            order = np.random.RandomState(_epoch_seed(caller.seed, epoch)).permutation(len(x))
            for begin in range(0, len(x), caller.recipe["batch_size"]):
                caller.update(caller.accumulate(x[order[begin:begin + 4]]))
            caller.end_epoch()
        result = caller.finalize()
        self.assertIs(result, caller.centroids)
        np.testing.assert_array_equal(result, matrix.centroids)
        np.testing.assert_array_equal(caller.obj, matrix.obj)
        self.assertEqual(caller.n_splits, matrix.n_splits)

    def test_lifecycle_rejects_incomplete_batches(self):
        km = Emeans(1, 1, niter=1, batch_size=3, alpha0=0.5)
        with self.assertRaises(RuntimeError):
            km.accumulate(np.zeros((3, 1), dtype="float32"))
        km.begin(5, np.zeros((1, 1), dtype="float32"))
        with self.assertRaises(ValueError):
            km.accumulate(np.zeros((4, 1), dtype="float32"))
        with self.assertRaises(RuntimeError):
            km.finalize()
        empty = km.accumulate(np.empty((0, 1), dtype="float32"))
        with self.assertRaises(ValueError):
            km.update(empty)
        km.reset()
        km.begin(5, np.zeros((1, 1), dtype="float32"))
        stats = km.accumulate(np.zeros((1, 1), dtype="float32"))
        self.assertEqual(stats["n"], 1)
        self.assertIsNone(stats["sums"])
        self.assertIsNone(stats["counts"])
        with self.assertRaises(ValueError):
            km.update(stats)
        empty = km.accumulate(np.empty((0, 1), dtype="float32"))
        self.assertEqual(empty["n"], 1)
        self.assertIsNone(empty["sums"])
        with self.assertRaises(ValueError):
            km.accumulate(np.zeros((3, 1), dtype="float32"))
        stats = km.accumulate(np.zeros((2, 1), dtype="float32"))
        self.assertEqual(stats["n"], 3)
        empty = km.accumulate(np.empty((0, 1), dtype="float32"))
        np.testing.assert_array_equal(empty["sums"], stats["sums"])
        np.testing.assert_array_equal(empty["counts"], stats["counts"])
        # A completed positive contribution still requires consumption.
        with self.assertRaises(RuntimeError):
            km.accumulate(np.zeros((3, 1), dtype="float32"))
        km.update(stats)
        with self.assertRaises(RuntimeError):
            km.end_epoch()
        stats = km.accumulate(np.ones((2, 1), dtype="float32"))
        km.update(stats)
        with self.assertRaises(RuntimeError):
            km.accumulate(np.ones((2, 1), dtype="float32"))
        with self.assertRaises(RuntimeError):
            km.update(stats)
        km.end_epoch()
        with self.assertRaises(RuntimeError):
            km.update(stats)
        np.testing.assert_array_equal(
            km.finalize(), np.array([[0.4]], dtype="float32")
        )

    def test_revival_starts_on_fifth_update(self):
        x = np.zeros((10, 1), dtype="float32")
        init = np.array([[0.0], [10.0]], dtype="float32")

        before = Emeans(1, 2, niter=4, batch_size=10, alpha0=0.5)
        before.train(x, init_centroids=init)
        self.assertEqual(before.n_splits, 0)

        on_fifth = Emeans(1, 2, niter=5, batch_size=10, alpha0=0.5)
        on_fifth.train(x, init_centroids=init)
        self.assertEqual(on_fifth.n_splits, 1)

    def test_revival_follows_current_batch_update(self):
        seed = 7
        order = np.random.RandomState(_epoch_seed(seed, 0)).permutation(5)
        x = np.zeros((5, 1), dtype="float32")
        x[order[-1]] = 10.0
        init = np.array([[0.0], [10.0]], dtype="float32")
        km = Emeans(
            1, 2, niter=1, batch_size=1, alpha0=0.5, seed=seed
        )

        km.train(x, init_centroids=init)

        # The fifth batch makes cluster 1 live before revival checks usage.
        self.assertEqual(km.n_splits, 0)

    def test_each_pass_uses_tail(self):
        x = np.array([[0.0], [0.0], [0.0], [0.0], [10.0]], dtype="float32")
        km = Emeans(1, 1, niter=2, batch_size=3, alpha0=0.5)

        km.train(x, init_centroids=np.array([[0.0]], dtype="float32"))

        np.testing.assert_array_equal(km.centroids, [[2.0]])
        self.assertEqual(km.recipe["steps_per_pass"], 2)
        self.assertEqual(km.recipe["total_steps"], 4)
        self.assertEqual(km.recipe["tail_policy"], "keep")

    def test_assignment_chunking_preserves_logical_updates(self):
        rng = np.random.RandomState(123)
        x = rng.randn(41, 5).astype("float32")
        init = x[:4].copy()
        args = {"niter": 3, "batch_size": 13, "alpha0": 0.2, "seed": 7}
        a = Emeans(5, 4, assignment_chunk_size=4, **args)
        b = Emeans(5, 4, assignment_chunk_size=64, **args)

        a.train(x, init_centroids=init)
        b.train(x, init_centroids=init)

        # This fixture remains away from assignment boundaries. In general,
        # Faiss may choose different exact-search kernels for different query
        # sizes, so chunking is not promised to be bitwise invariant.
        np.testing.assert_allclose(a.centroids, b.centroids)
        np.testing.assert_allclose(a.obj, b.obj)
        self.assertEqual(a.recipe["steps_per_pass"], 4)
        self.assertEqual(a.recipe["total_steps"], 12)

    def test_retrain_is_deterministic_and_reset(self):
        x = np.random.RandomState(123).randn(40, 4).astype("float32")
        km = Emeans(4, 3, niter=2, batch_size=11, alpha0=0.2)
        km.train(x)
        centroids = km.centroids.copy()
        objective = km.obj.copy()

        km.train(x)

        np.testing.assert_array_equal(km.centroids, centroids)
        np.testing.assert_array_equal(km.obj, objective)
        km.reset()
        with self.assertRaises(RuntimeError):
            km.assign(x)

    def test_training_does_not_advance_global_rng(self):
        x = np.arange(80, dtype="float32").reshape(20, 4)
        state = np.random.get_state()
        km = Emeans(4, 3, niter=1, batch_size=7, alpha0=0.2)

        km.train(x)

        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_invalid_input(self):
        km = Emeans(3, 2, niter=1, batch_size=2, alpha0=0.5)
        with self.assertRaises(ValueError):
            km.train(np.zeros((4, 2), dtype="float32"))
        with self.assertRaises(ValueError):
            km.train(np.full((4, 3), np.nan, dtype="float32"))
        with self.assertRaises(RuntimeError):
            km.assign(np.zeros((1, 3), dtype="float32"))

if __name__ == "__main__":
    unittest.main()
