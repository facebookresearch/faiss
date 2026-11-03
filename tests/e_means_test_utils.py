# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared NumPy/native-Torch reference contracts; no optional imports at load."""
import numpy as np
from contextlib import contextmanager
from unittest import mock
import weakref

from faiss.contrib.e_means import Emeans
from faiss.contrib.e_means import e_means


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def known_torch_centroids(centroids, **kwargs):
    """Install fixed means through the documented external-statistics API."""
    import torch
    k, d = centroids.shape
    model = Emeans(d, k, niter=1, batch_size=k, alpha0=1, **kwargs)
    model.begin(k, centroids)
    model.update(dict(sums=centroids.clone(), counts=torch.ones(k, device=centroids.device),
                      objective=torch.zeros((), dtype=torch.float64, device=centroids.device), n=k))
    model.end_epoch()
    torch.testing.assert_close(model.finalize(), centroids, rtol=0, atol=0)
    return model


@contextmanager
def highest_fp32(backend):
    if backend == "numpy":
        yield
        return
    import torch
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


class EmeansReferenceContracts:
    """The two public test modules supply their own array/device constructors."""

    def test_spherical_ip_zero_objective_and_represented_statistics(self):
        initial = np.array([[0, 0], [1, 0], [-1, 0]], dtype="float32")
        rows = np.array([[0, 0], [.25, np.sqrt(15)/4], [1, 0], [-1, 0],
                         [0, 1], [0, -1]], dtype="float32")
        expected_labels = np.array([0, 1, 1, 2, 0, 0])
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend), highest_fp32(backend):
                model = Emeans(2, 3, niter=1, batch_size=6, alpha0=1,
                               spherical=True, capture_assignments=True, assignment_chunk_size=2)
                source = array(initial)
                model.begin(6, source)
                stats = model.accumulate(array(rows))
                np.testing.assert_array_equal(as_numpy(stats["labels"]), expected_labels)
                counts = np.bincount(expected_labels, minlength=3)
                sums = np.zeros_like(initial)
                np.add.at(sums, expected_labels, rows)
                np.testing.assert_array_equal(as_numpy(stats["counts"]), counts)
                np.testing.assert_array_equal(as_numpy(stats["sums"]), sums)
                self.assertAlmostEqual(float(stats["objective"]), 7.5, places=6)
                model.update(stats)
                model.end_epoch()
                final = as_numpy(model.finalize())
                np.testing.assert_array_equal(final[0], [0, 0])
                similarity, labels = model.assign(array(rows))
                scores = rows @ final.T
                np.testing.assert_array_equal(as_numpy(labels), scores.argmax(axis=1))
                np.testing.assert_allclose(as_numpy(similarity), scores.max(axis=1), atol=2e-7)
                np.testing.assert_array_equal(as_numpy(source), initial)
                self.assertAlmostEqual(float(as_numpy(model.obj)[0]), 7.5, places=6)

    def test_spherical_deferred_ema_borrowed_cache_and_chunks(self):
        rows = np.array([[1, 0], [0, 1], [-1, 0]], dtype="float32")
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend), highest_fp32(backend):
                args = dict(niter=1, batch_size=3, alpha0=.25, spherical=True)
                model, control = Emeans(2, 1, **args), Emeans(2, 1, **args)
                for km in (model, control):
                    km.begin(6, array([[3, 4]]))
                raw = as_numpy(model._centroids).copy()
                usage = as_numpy(model._usage).copy()
                handle = model.current_centroids()
                cache = model._unit_centroids
                version = getattr(cache, "_version", None)
                before = as_numpy(handle).copy()
                model.accumulate(array(rows[:1]))
                for _ in range(3):
                    np.testing.assert_array_equal(as_numpy(model.current_centroids()), before)
                    if backend != "numpy":
                        model._prepare_torch_assignment(model._centroids)
                stats = model.accumulate(array(rows[1:]))
                self.assertIs(model._unit_centroids, cache)
                self.assertEqual(getattr(cache, "_version", None), version)
                np.testing.assert_array_equal(as_numpy(handle), before)
                self.assertAlmostEqual(float(stats["objective"]), 4.4, places=5)
                mass = .75 * usage + .25 * 3
                expected = raw + (.25 * 3 / mass)[:, None] * (rows.mean(axis=0) - raw)
                model.update(stats)
                control.update(control.accumulate(array(rows)))
                np.testing.assert_allclose(as_numpy(model._centroids), expected, atol=1e-7)
                self.assertLess(np.linalg.norm(as_numpy(model._centroids)[0]), .999)
                np.testing.assert_allclose(
                    as_numpy(model.current_centroids()),
                    expected / np.linalg.norm(expected, axis=1)[:, None],
                    atol=1e-7,
                )
                self.assertIs(model._unit_centroids, cache)
                for km in (model, control):
                    km.update(km.accumulate(array(rows)))
                    km.end_epoch()
                np.testing.assert_array_equal(as_numpy(model.finalize()), as_numpy(control.finalize()))
                np.testing.assert_array_equal(as_numpy(model.obj), as_numpy(control.obj))

    def test_spherical_projection_extremes_and_wide_closing(self):
        from faiss.contrib.e_means.e_means_spherical import normalize_into

        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend):
                source = array([[3e38, -3e38], [1e-38, 0], [0, 0]])
                saved = as_numpy(source).copy()
                output = array(np.empty((3, 2), dtype="float32"))
                normalize_into(source, output)
                np.testing.assert_allclose(
                    as_numpy(output),
                    [[2**-.5, -2**-.5], [1, 0], [0, 0]],
                    atol=1e-7,
                )
                np.testing.assert_array_equal(as_numpy(source), saved)
                model = Emeans(2, 3, niter=1, batch_size=3, alpha0=1, spherical=True)
                model.begin(3, source)
                np.testing.assert_allclose(as_numpy(model.current_centroids()), as_numpy(output), atol=1e-7)
                model.begin(3, array([[1, 0], [0, 1], [-1, 0]]))
                model.update(model.accumulate(array([[1, 0], [0, 1], [-1, 0]])))
                model.end_epoch()
                # Direct finalize seam: no huge dataset; mean must project before F32.
                model._pass_sums = array([[1e300, -1e300], [0, 0], [0, 0]], dtype="float64")
                model._pass_counts = array([1, 1, 0], dtype="float64")
                np.testing.assert_allclose(
                    as_numpy(model.finalize()),
                    [[2**-.5, -2**-.5], [0, 0], [-1, 0]],
                    atol=1e-7,
                )
                self.assertIsNone(model._unit_centroids)

    def test_spherical_zero_beats_negative_similarity(self):
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend):
                model = Emeans(2, 2, niter=1, batch_size=2, alpha0=1, spherical=True,
                               capture_assignments=True)
                model.begin(2, array([[0, 0], [1, 0]]))
                stats = model.accumulate(array([[-1, 0], [0, 0]]))
                np.testing.assert_array_equal(as_numpy(stats["labels"]), [0, 0])
                self.assertEqual(float(stats["objective"]), 4)

    def test_spherical_torch_version_and_inference_invalidation(self):
        for backend, array in self.reference_arrays():
            if backend == "numpy":
                continue
            import torch

            for inference in (False, True):
                with self.subTest(backend=backend, inference=inference), torch.inference_mode(inference):
                    model = Emeans(2, 1, niter=1, batch_size=2, alpha0=1, spherical=True)
                    model.begin(2, array([[1, 0]]))
                    before = model.current_centroids()
                    cache = model._unit_centroids
                    model.current_centroids()
                    self.assertIs(model._unit_centroids, cache)
                    if not inference:
                        # Version counters also catch ATen raw writes, independent of hooks.
                        model._centroids.copy_(array([[0, 2]]))
                        np.testing.assert_array_equal(as_numpy(model.current_centroids()), [[0, 1]])
                    model.update(model.accumulate(array([[0, -1], [0, -1]])))
                    np.testing.assert_array_equal(as_numpy(model.current_centroids()), [[0, -1]])
                    self.assertIs(model._unit_centroids, cache)
                    self.assertTrue(torch.isfinite(before).all())
                    model.end_epoch()
                    model.finalize()

    def test_spherical_fractional_caller_and_matrix(self):
        rng = np.random.RandomState(73)
        rows = rng.randn(31, 5).astype("float32")
        rows /= np.linalg.norm(rows, axis=1)[:, None]
        for backend, array in self.reference_arrays():
            for budget in (.5, 1.5):
                with self.subTest(backend=backend, budget=budget), highest_fp32(backend):
                    args = dict(niter=budget, spherical=True, batch_size=7, alpha0=.2,
                                seed=7, capture_assignments=True, assignment_chunk_size=3)
                    matrix, caller = Emeans(5, 4, **args), Emeans(5, 4, **args)
                    matrix.train(array(rows), array(rows[:4]))
                    caller.begin(31, array(rows[:4]))
                    labels = np.full(31, -1, dtype="int64")
                    for epoch, n in enumerate(caller.recipe["epoch_rows"]):
                        from faiss.contrib.e_means.e_means import _epoch_seed, _torch_generator
                        seed = _epoch_seed(7, epoch)
                        if backend == "numpy":
                            order = np.random.RandomState(seed).permutation(31)
                        else:
                            import torch
                            x = array(rows)
                            order = torch.randperm(
                                31,
                                device=x.device,
                                generator=_torch_generator(x.device, seed),
                            ).cpu().numpy()
                        for begin in range(0, n, 7):
                            ids = order[begin:min(n, begin + 7)]
                            stats = caller.accumulate(array(rows[ids]))
                            if epoch == len(caller.recipe["epoch_rows"]) - 1:
                                labels[ids] = as_numpy(stats["labels"])
                            caller.update(stats)
                        caller.end_epoch()
                    np.testing.assert_allclose(as_numpy(matrix.centroids), as_numpy(caller.finalize()), atol=2e-7)
                    np.testing.assert_array_equal(as_numpy(matrix.assignments), labels)
                    np.testing.assert_allclose(as_numpy(matrix.obj), as_numpy(caller.obj), atol=2e-6)

    def test_fractional_epochs_preserve_batches_and_closing_labels(self):
        cases = ((2, [31, 31]), (2.5, [31, 16, 31]),
                 (1.25, [8, 31]), (0.5, [16]), (0.1, [3]))
        for backend, array in self.reference_arrays():
            for budget, epoch_rows in cases:
                with self.subTest(backend=backend, budget=budget), highest_fp32(backend):
                    x = array(np.arange(31, dtype='float32')[:, None] / 4)
                    initial = x[[0, 10, 20, 30]]
                    args = dict(niter=budget, batch_size=7, alpha0=0.2, seed=7,
                                assignment_chunk_size=3, capture_assignments=True)
                    matrix, caller = Emeans(1, 4, **args), Emeans(1, 4, **args)
                    matrix.train(x, initial)
                    caller.begin(len(x), initial)
                    steps = [(n + 6) // 7 for n in epoch_rows]
                    self.assertEqual(matrix.recipe['epoch_rows'], epoch_rows)
                    self.assertEqual(matrix.recipe['epoch_steps'], steps)
                    self.assertEqual(matrix.recipe['total_steps'], sum(steps))
                    for epoch, rows in enumerate(epoch_rows):
                        seed = e_means._epoch_seed(7, epoch)
                        if backend == 'numpy':
                            order = np.random.RandomState(seed).permutation(len(x))
                        else:
                            import torch
                            generator = torch.Generator(device=x.device).manual_seed(seed)
                            order = torch.randperm(len(x), generator=generator, device=x.device)
                        for start in range(0, rows, 7):
                            batch = x[order[start:min(start + 7, rows)]]
                            frozen = as_numpy(caller.current_centroids()).copy()
                            stats = caller.accumulate(batch[:2])
                            if len(batch) > 2:
                                stats = caller.accumulate(batch[2:])
                            np.testing.assert_array_equal(as_numpy(caller.current_centroids()), frozen)
                            caller.update(stats)
                        self.assertEqual(caller._epoch_steps, steps[epoch])
                        caller.end_epoch()
                    self.assertEqual(matrix._step, sum(steps))
                    np.testing.assert_allclose(as_numpy(caller.finalize()), as_numpy(matrix.centroids),
                                               atol=1e-6, rtol=1e-6)
                    np.testing.assert_allclose(as_numpy(caller.obj), as_numpy(matrix.obj),
                                               atol=1e-5, rtol=1e-6)
                    captured = as_numpy(matrix.assignments)
                    visited = as_numpy(order)[:epoch_rows[-1]]
                    absent = np.ones(len(x), dtype=bool)
                    absent[visited] = False
                    np.testing.assert_array_equal(captured[absent], -1)
                    self.assertTrue(np.all(captured[visited] >= 0))
                    for centroid in range(4):
                        assigned = visited[captured[visited] == centroid]
                        if len(assigned):
                            expected = as_numpy(x)[assigned].astype('float64').mean(0)
                            np.testing.assert_allclose(as_numpy(matrix.centroids)[centroid], expected,
                                                       atol=1e-6, rtol=1e-6)
                    counts = np.bincount(captured[visited], minlength=4).astype('float64')
                    self.assertEqual(matrix.iteration_stats[-1]['imbalance_factor'],
                                     4 * counts.dot(counts) / len(visited)**2)

    def test_supplied_chunks_share_one_frozen_batch(self):
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend), highest_fp32(backend):
                rows = array([[0.25, 0.0], [10.0, 10.25], [0.5, 0.25],
                              [10.25, 10.0], [0.0, 0.25], [10.5, 10.25],
                              [0.5, 0.0], [10.0, 10.5], [0.25, 0.5],
                              [10.25, 10.5], [0.5, 0.5], [10.5, 10.5], [0.0, 0.0]])
                initial = array([[0.0, 0.0], [10.0, 10.0]])
                args = dict(niter=2, batch_size=4, alpha0=0.3, seed=7,
                            assignment_chunk_size=2, capture_assignments=True)
                whole, chunked = Emeans(2, 2, **args), Emeans(2, 2, **args)
                whole.begin(len(rows), initial)
                chunked.begin(len(rows), initial)
                empty = chunked.accumulate(rows[:0])
                self.assertEqual(empty["n"], 0)
                np.testing.assert_array_equal(as_numpy(empty["counts"]), [0, 0])
                saved_labels = []
                for epoch in range(2):
                    for start in range(0, len(rows), 4):
                        batch = rows[start:start+4]
                        reference = whole.accumulate(batch)
                        frozen = as_numpy(chunked.current_centroids()).copy()
                        count = 0
                        for offset in range(0, len(batch), 2):
                            part = batch[offset:offset+2]
                            stats = chunked.accumulate(part)
                            count += len(part)
                            self.assertEqual(stats["n"], count)
                            np.testing.assert_array_equal(as_numpy(chunked.current_centroids()), frozen)
                            self.assertEqual(chunked._step, whole._step)
                            if count < len(batch):
                                self.assertIsNone(stats["sums"])
                                self.assertIsNone(stats["counts"])
                                # Host errors must preserve the partial batch,
                                # and reject before inspecting replacement fields.
                                with mock.patch.object(chunked, "_accumulate_core", side_effect=AssertionError):
                                    with self.assertRaises(ValueError):
                                        chunked.accumulate(rows)
                                    with self.assertRaises(ValueError):
                                        chunked.update({})
                                with self.assertRaises(RuntimeError):
                                    chunked.end_epoch()
                                with self.assertRaises(RuntimeError):
                                    chunked.finalize()
                            if epoch == 1:
                                labels = stats["labels"]
                                expected = ((as_numpy(part).astype("float64")[:, None, :] -
                                             frozen.astype("float64")[None, :, :])**2).sum(axis=2).argmin(axis=1)
                                np.testing.assert_array_equal(as_numpy(labels), expected)
                                self.assertEqual(as_numpy(labels).dtype, np.dtype("int64"))
                                if backend != "numpy":
                                    self.assertEqual(labels.device, rows.device)
                                    self.assertFalse(labels.requires_grad)
                                saved_labels.append((labels, as_numpy(labels).copy()))
                            else:
                                self.assertNotIn("labels", stats)
                            empty = chunked.accumulate(rows[:0])
                            self.assertEqual(empty["n"], count)
                            if count < len(batch):
                                self.assertIsNone(empty["sums"])
                                self.assertIsNone(empty["counts"])
                            else:
                                for field in ("sums", "counts", "objective"):
                                    np.testing.assert_array_equal(as_numpy(empty[field]), as_numpy(stats[field]))
                            if epoch == 1:
                                self.assertEqual(len(empty["labels"]), 0)
                        for field in ("sums", "counts", "objective"):
                            np.testing.assert_allclose(as_numpy(stats[field]), as_numpy(reference[field]),
                                                       rtol=1e-6, atol=1e-6)
                        with self.assertRaises(RuntimeError):
                            chunked.accumulate(batch[:1])
                        whole.update(reference)
                        chunked.update(stats)
                        for labels, snapshot in saved_labels:
                            np.testing.assert_array_equal(as_numpy(labels), snapshot)
                    whole.end_epoch()
                    chunked.end_epoch()
                np.testing.assert_allclose(as_numpy(chunked.finalize()), as_numpy(whole.finalize()),
                                           rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(as_numpy(chunked.obj), as_numpy(whole.obj), rtol=1e-6, atol=1e-6)
                self.assertEqual(chunked.n_splits, whole.n_splits)
                self.assertIsNone(chunked.assignments)

    def test_last_pass_labels_are_owned_and_not_final_nearest(self):
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend), highest_fp32(backend):
                rows = array([[0.0], [1.0], [49.0], [51.0], [52.0], [53.0]])
                initial = array([[0.0], [100.0]])
                args = dict(niter=1, batch_size=6, alpha0=0.5, seed=7)
                model = Emeans(1, 2, capture_assignments=True, **args)
                self.assertIsNone(model.assignments)
                model.train(rows, initial)
                labels = model.assignments
                np.testing.assert_array_equal(as_numpy(labels), [0, 0, 0, 1, 1, 1])
                self.assertEqual(as_numpy(labels).dtype, np.dtype("int64"))
                np.testing.assert_allclose(as_numpy(model.centroids), [[50/3], [52]], rtol=1e-6)
                _, nearest = model.assign(rows)
                self.assertEqual(int(nearest[2]), 1)
                self.assertEqual(int(labels[2]), 0)
                disabled = Emeans(1, 2, capture_assignments=False, **args)
                disabled.train(rows, initial)
                self.assertIsNone(disabled.assignments)
                np.testing.assert_array_equal(as_numpy(model.centroids), as_numpy(disabled.centroids))
                np.testing.assert_array_equal(as_numpy(model.obj), as_numpy(disabled.obj))
                # Old caller-owned output survives, but a failed new attempt
                # must not present it as the new train's result.
                with self.assertRaises(ValueError):
                    model.train(array([[0.0, 0.0]]))
                self.assertIsNone(model.assignments)
                np.testing.assert_array_equal(as_numpy(labels), [0, 0, 0, 1, 1, 1])
                model.begin(6, initial)
                model.accumulate(rows[:2])
                model.reset()
                self.assertIsNone(model.assignments)
                model.train(rows, initial)
                np.testing.assert_array_equal(as_numpy(model.assignments), as_numpy(labels))

    def test_pending_execution_failure_requires_new_run(self):
        class InjectedFailure(BaseException):
            pass

        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend):
                # The view forces real input preparation for the three-row
                # continuation rather than testing only contiguous aliases.
                rows = array([[0.0, 99.0], [2.0, 99.0], [10.0, 99.0], [12.0, 99.0]])[:, :1]
                initial = rows[[0, 2]]
                model = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5,
                               capture_assignments=True)
                for stage in ("assignment", "preparation"):
                    with self.subTest(stage=stage):
                        model.begin(4, initial)
                        first = model.accumulate(rows[:1])
                        labels, saved = first["labels"], as_numpy(first["labels"]).copy()
                        with self.assertRaises(ValueError):
                            model.accumulate(array([[0.0, 1.0]]))
                        if backend in ("numpy", "cpu"):
                            with self.assertRaises(ValueError):
                                model.accumulate(array([[float("nan")]]))
                        owner = model if stage == "assignment" else e_means
                        name = "_assign_rows" if stage == "assignment" else "_prepare_float32_matrix"
                        original = getattr(owner, name)

                        def fail_after_work(*args, _original=original):
                            _original(*args)
                            raise InjectedFailure("after tensor work")

                        with mock.patch.object(owner, name, side_effect=fail_after_work):
                            with self.assertRaises(InjectedFailure):
                                model.accumulate(rows[1:])
                        self.assertIsNone(model.assignments)
                        for operation in (
                            lambda _model=model, _rows=rows: _model.accumulate(_rows),
                            lambda _model=model: _model.update({}),
                            model.end_epoch,
                            model.finalize,
                            model.current_centroids,
                        ):
                            with self.assertRaises(RuntimeError):
                                operation()
                        np.testing.assert_array_equal(as_numpy(labels), saved)
                        model.begin(4, initial)
                        model.update(model.accumulate(rows))
                        model.end_epoch()
                        np.testing.assert_array_equal(as_numpy(model.finalize()), [[1.0], [11.0]])

    def test_external_statistics_schema_and_result_ownership(self):
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend):
                initial = array([[0.0], [10.0]])
                model = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)
                with self.assertRaises(RuntimeError):
                    model.current_centroids()
                model.begin(4, initial)
                initial[...] = 7
                self.assertIsNone(model.centroids)
                current = model.current_centroids()
                np.testing.assert_array_equal(as_numpy(current), [[0.0], [10.0]])
                self.assertEqual(tuple(current.shape), (2, 1))
                if backend == "numpy":
                    import faiss

                    self.assertEqual(current.dtype, np.dtype("float32"))
                    self.assertTrue(np.shares_memory(current, model.current_centroids()))
                    self.assertFalse(current.flags.writeable)
                    with self.assertRaises(ValueError):
                        current[0, 0] = 1
                    distance, labels = faiss.knn(array([[0.0], [10.0]]), current, 1)
                    np.testing.assert_array_equal(distance, [[0.0], [0.0]])
                    np.testing.assert_array_equal(labels, [[0], [1]])
                    snapshot = current.copy()
                else:
                    self.assertEqual(current.dtype, initial.dtype)
                    self.assertEqual(current.device, initial.device)
                    self.assertFalse(current.requires_grad)
                    self.assertIsNone(current.grad_fn)
                    self.assertEqual(current.data_ptr(), model.current_centroids().data_ptr())
                    snapshot = current.clone()
                empty = model.accumulate(array(np.empty((0, 1))))
                self.assertEqual(empty["n"], 0)
                np.testing.assert_array_equal(as_numpy(empty["counts"]), [0, 0])
                with self.assertRaises(ValueError):
                    model.update(empty)
                with self.assertRaises(ValueError):
                    model.accumulate(array(np.ones((5, 1))))
                # Closing leaves one cluster untouched: a begin alias cannot
                # be masked by overwriting every centroid with new means.
                combined = dict(sums=array([[8.0], [0.0]]), counts=array([4.0, 0.0]),
                                objective=array(8.0, "float64"), n=4)
                for field in ("sums", "counts", "objective"):
                    wrong = dict(combined)
                    wrong[field] = array(as_numpy(combined[field]),
                                         "float32" if field == "objective" else "float64")
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        model.update(wrong)
                model.update(combined)
                expected = np.float32(8.0 / (4.0 + 2**-16))
                np.testing.assert_allclose(as_numpy(model.current_centroids()),
                                           [[expected], [10.0]], rtol=2e-7)
                np.testing.assert_array_equal(as_numpy(snapshot), [[0.0], [10.0]])
                self.assertIsNone(model.centroids)
                before_end = as_numpy(model.current_centroids()).copy()
                model.end_epoch()
                np.testing.assert_array_equal(as_numpy(model.current_centroids()), before_end)
                result = model.finalize()
                with self.assertRaises(RuntimeError):
                    model.current_centroids()
                np.testing.assert_array_equal(as_numpy(result), [[2.0], [10.0]])
                self.assertIs(model.finalize(), result)
                model.reset()
                with self.assertRaises(RuntimeError):
                    model.current_centroids()
                np.testing.assert_array_equal(as_numpy(result), [[2.0], [10.0]])
                with self.assertRaises(RuntimeError):
                    model.assign(array([[0.0]]))
                model.begin(4, array([[3.0], [13.0]]))
                np.testing.assert_array_equal(as_numpy(model.current_centroids()), [[3.0], [13.0]])
                rows = array([[3.0], [5.0], [3.0], [5.0]])
                model.accumulate(rows[:0])
                partial = model.accumulate(rows[:2])
                self.assertEqual(partial["n"], 2)
                self.assertIsNone(partial["sums"])
                self.assertIsNone(partial["counts"])
                replacement = dict(sums=array([[16.0], [0.0]]), counts=array([4.0, 0.0]),
                                   objective=array(8.0, "float64"), n=4)
                with self.assertRaises(ValueError):
                    model.update(replacement)
                complete = model.accumulate(rows[2:])
                empty = model.accumulate(rows[:0])
                for field in ("sums", "counts", "objective"):
                    np.testing.assert_array_equal(as_numpy(empty[field]), as_numpy(complete[field]))
                copied = {field: e_means._clone(empty[field])
                          for field in ("sums", "counts", "objective")}
                copied["n"] = empty["n"]
                model.update(copied)
                model.end_epoch()
                np.testing.assert_array_equal(as_numpy(model.finalize()), [[4.0], [13.0]])

    def test_fractional_near_tie_selected_distance(self):
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend), highest_fp32(backend):
                model = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5,
                               assignment_chunk_size=1, centroid_chunk_size=1)
                model.train(array([[0.0], [0.0], [1.0], [1.0]]), array([[0.0], [1.0]]))
                query = array([[0.5+2**-12], [0.5-2**-12], [0.5]])
                distance, labels = model.assign(query)
                direct = (as_numpy(query).astype("float64") -
                          as_numpy(model.centroids).astype("float64").T)**2
                np.testing.assert_array_equal(as_numpy(labels), [1, 0, 0])
                # Dyadic inputs and a resolved margin, not a requirement that
                # arbitrary near-ties survive expanded-L2 cancellation.
                np.testing.assert_allclose(as_numpy(distance), direct.min(axis=1),
                                           rtol=0, atol=1e-7)

    def test_initial_mass_first_update_and_zero_mass_reentry(self):
        for backend, array in self.reference_arrays():
            for alpha, zero_mass, position in ((0.5, False, 3.0),
                                               (1.0, False, 1e8),
                                               (0.5, True, 1e8)):
                with self.subTest(backend=backend, alpha=alpha, zero_mass=zero_mass):
                    model = Emeans(1, 2, niter=2, batch_size=2, alpha0=alpha)
                    model.begin(2, array([[position], [8.25]]))
                    self.assertEqual(model.recipe["init_usage"], 2**-16)
                    # First-step state is not exposed before closing. Inject
                    # zero mass only to isolate the documented reentry case.
                    if zero_mass:
                        model._usage[0] = 0
                    mass = 0.0 if zero_mass else 2**-16
                    count = 2.0
                    weight = alpha*count / ((1-alpha)*mass + alpha*count)
                    expected = 1.0 if weight == 1 else position + weight*(1-position)
                    model.update(dict(sums=array([[2.0], [0.0]]),
                                      counts=array([2.0, 0.0]),
                                      objective=array(7.0, "float64"), n=2))
                    np.testing.assert_allclose(as_numpy(model._centroids),
                                               [[expected], [8.25]], rtol=2e-7)
                    unused_mass = 2**-16 if alpha == 1 else (1-alpha)*2**-16
                    self.assertEqual(float(model._usage[1]), unused_mass)
                    model.reset()

    def test_untouched_underflow_and_independent_ema(self):
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend):
                model = Emeans(1, 2, niter=1, batch_size=1, alpha0=0.9)
                model.begin(400, array([[0.0], [8.25]]))
                mass, position = 2**-16, 0.0
                # Disable revival only to isolate decay/EMA; separate tests
                # exercise real revival and closing without this probe.
                with mock.patch.object(model, "_revive_unused", return_value=0):
                    for step in range(400):
                        value = (step % 7 - 3) / 8
                        alpha = e_means._scheduled_alpha(0.9, step, 400)
                        mass = (1-alpha)*mass + alpha
                        position += alpha/mass*(value-position)
                        model.update(dict(sums=array([[value], [0.0]]),
                                          counts=array([1.0, 0.0]),
                                          objective=array(0.0, "float64"), n=1))
                        # Retained bounded fixture's FP64 scalar oracle and
                        # tolerance, not a new universal rounding guarantee.
                        np.testing.assert_allclose(float(model._centroids[0, 0]),
                                                   position, rtol=3e-6, atol=3e-7)
                        self.assertEqual(float(model._centroids[1, 0]), 8.25)
                self.assertLessEqual(float(model._usage[1]), np.finfo("float32").tiny)
                model.end_epoch()
                self.assertEqual(float(model.finalize()[1, 0]), 8.25)

    def test_revival_threshold_separation_and_conservation(self):
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend):
                model = Emeans(1, 3)
                model.begin(3, array([[10.0], [20.0], [30.0]]))
                # Controlled masses make the otherwise private revival
                # threshold/conservation boundary directly observable.
                model._usage[:] = array([27.715767, 0.38280505, 803.37445])
                before = as_numpy(model._usage).copy()
                coefficient = np.float32(model.recipe["unused_threshold"])
                expected = np.float32(np.float32(before.sum(dtype="float32")*coefficient)/3)
                large = array([69938690.0])
                if backend == "numpy":
                    actual = e_means._numpy_unused_threshold(before, 3)
                    large_threshold = e_means._numpy_unused_threshold(large, 52359)
                else:
                    # Exercise Torch's own reduction/multiply/divide path,
                    # rather than repeating the NumPy helper for CUDA inputs.
                    actual = model._usage.sum()*model.recipe["unused_threshold"]/model.k
                    large_threshold = large.sum()*model.recipe["unused_threshold"]/52359
                self.assertEqual(as_numpy(actual).dtype, np.dtype("float32"))
                self.assertEqual(float(actual), expected)
                self.assertEqual(int(as_numpy(actual).view("uint32")), 1105050084)
                self.assertEqual(as_numpy(large_threshold).dtype, np.dtype("float32"))
                self.assertEqual(int(as_numpy(large_threshold).view("uint32")), 1124438855)
                np.testing.assert_array_equal(before < float(actual), [False, True, False])
                weighted = (as_numpy(model._centroids)*before[:, None]).sum(axis=0)
                discarded = as_numpy(model._centroids)[1]*before[1]
                if backend == "numpy":
                    rng = np.random.RandomState(123)
                else:
                    import torch
                    rng = torch.Generator(device=model._usage.device).manual_seed(123)
                self.assertEqual(model._revive_unused(rng), 1)
                # Infer the sampled donor rather than pinning CPU/CUDA RNG.
                # Ordinary revival retains the documented 1e-4 displacement.
                after = as_numpy(model._usage)
                source = np.flatnonzero((before != after) & (np.arange(3) != 1))
                self.assertEqual(len(source), 1)
                source = int(source[0])
                initial_position = np.float32([10.0, 20.0, 30.0][source])
                state = as_numpy(model._centroids)
                np.testing.assert_allclose(state[source], [initial_position*np.float32(0.9999)], rtol=1e-6, atol=0)
                np.testing.assert_allclose(state[1], [initial_position*np.float32(1.0001)], rtol=1e-6, atol=0)
                other = 2 if source == 0 else 0
                np.testing.assert_array_equal(state[other], [[10.0], [20.0], [30.0]][other])
                np.testing.assert_allclose(as_numpy(model._usage).sum(), before.sum()-before[1])
                np.testing.assert_allclose(
                    (as_numpy(model._centroids)*as_numpy(model._usage)[:, None]).sum(axis=0),
                    weighted-discarded)
                donor = array([[0.0, 0.0], [2.0, 3.0], [1.0, 0.0]])
                child = array([[0.0, 0.0], [2.0002, 2.9997], [1.0, 0.0]])
                ordinary = as_numpy(child)[1].copy()
                e_means._separate_splits(donor, child)
                result = as_numpy(child)
                self.assertEqual(result[0, 0], np.nextafter(np.float32(0), np.float32(np.inf)))
                self.assertEqual(result[2, 0], np.nextafter(np.float32(1), np.float32(np.inf)))
                np.testing.assert_array_equal(result[1], ordinary)
                # One donor and two unused clusters: at most one is revived,
                # the donor/child separate, and the third is left untouched.
                unique = Emeans(2, 3)
                unique.begin(3, array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]))
                unique._usage[:] = array([10.0, 0.0, 0.0])
                self.assertEqual(unique._revive_unused(rng), 1)
                np.testing.assert_array_equal(as_numpy(unique._usage), [5.0, 5.0, 0.0])
                state = as_numpy(unique._centroids)
                self.assertFalse(np.array_equal(state[0], state[1]))
                np.testing.assert_array_equal(state[2], [20.0, 20.0])

    def test_closing_statistics_survive_revival_and_reset(self):
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend):
                model = Emeans(1, 2, niter=1, batch_size=1, alpha0=0.5)
                model.begin(5, array([[0.0], [100.0]]))
                # No caller holds these working allocations. Check release,
                # not cache keys or an optimization's storage identity.
                batch_owner = weakref.ref(model._batch_stats)
                closing_owner = weakref.ref(model._pass_sums)
                for value in range(5):
                    model.update(dict(sums=array([[float(value)], [0.0]]),
                                      counts=array([1.0, 0.0]),
                                      objective=array(1.0, "float64"), n=1))
                self.assertEqual(model.n_splits, 1)
                revived = float(model._centroids[1, 0])
                model.end_epoch()
                result = model.finalize()
                np.testing.assert_array_equal(as_numpy(result), [[2.0], [revived]])
                self.assertEqual(float(model.obj[0]), 5.0)
                self.assertEqual(model.iteration_stats[0]["imbalance_factor"], 2.0)
                self.assertIs(model.finalize(), result)
                before = as_numpy(result).copy()
                model.reset()
                np.testing.assert_array_equal(as_numpy(result), before)
                self.assertIsNone(batch_owner())
                self.assertIsNone(closing_owner())

    def test_fractional_caller_and_matrix_training(self):
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend):
                x = array(np.arange(51, dtype="float32").reshape(17, 3)/10)
                initial = x[[0, 5, 10, 16]]
                args = dict(niter=3, batch_size=4, alpha0=0.2, seed=7,
                            assignment_chunk_size=2, centroid_chunk_size=2)
                matrix, caller = Emeans(3, 4, **args), Emeans(3, 4, **args)
                matrix.train(x, initial)
                caller.begin(len(x), initial)
                for epoch in range(3):
                    seed = e_means._epoch_seed(7, epoch)
                    if backend == "numpy":
                        order = np.random.RandomState(seed).permutation(len(x))
                    else:
                        import torch
                        generator = torch.Generator(device=x.device).manual_seed(seed)
                        order = torch.randperm(len(x), generator=generator, device=x.device)
                    for begin in range(0, len(x), 4):
                        caller.update(caller.accumulate(x[order[begin:begin+4]]))
                    caller.end_epoch()
                np.testing.assert_allclose(as_numpy(caller.finalize()), as_numpy(matrix.centroids),
                                           rtol=1e-6, atol=1e-6)
                # Existing decimal-fixture limits account for the retained
                # FP32 addition/cancellation order; exact binary coverage stays.
                atol = 3e-5 if backend == "cuda" else 1e-5
                np.testing.assert_allclose(as_numpy(caller.obj), as_numpy(matrix.obj),
                                           rtol=1e-6, atol=atol)
                self.assertEqual(caller.n_splits, matrix.n_splits)
