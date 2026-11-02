# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
import numpy as np

from faiss.contrib.e_means import Emeans
from faiss.contrib.e_means.e_means import _epoch_seed
from faiss.contrib.e_means import e_means
from e_means_test_utils import EmeansReferenceContracts


class TestTorchEmeans(EmeansReferenceContracts, unittest.TestCase):
    def reference_arrays(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        return [(device, lambda values, dtype="float32", device=device:
                 torch.tensor(values, dtype=getattr(torch, dtype), device=device))
                for device in devices]

    def test_fractional_native_and_numpy_training(self):
        rng = np.random.RandomState(123)
        centers = (10*rng.randn(4, 6)).astype("float32")
        rows = centers[np.arange(80) % 4] + (0.1*rng.randn(80, 6)).astype("float32")
        args = dict(niter=3, batch_size=len(rows), alpha0=0.2, seed=7)
        reference = Emeans(6, 4, **args)
        reference.train(rows, centers)
        for backend, array in self.reference_arrays():
            with self.subTest(backend=backend):
                model = Emeans(6, 4, **args)
                model.train(array(rows), array(centers))
                np.testing.assert_allclose(model.centroids.cpu().numpy(), reference.centroids,
                                           atol=2e-5, rtol=1e-6)
                np.testing.assert_allclose(model.obj.cpu().numpy(), reference.obj,
                                           atol=3e-3, rtol=1e-3)
                self.assertEqual(model.n_splits, reference.n_splits)

    def test_cpu_rejects_nonfinite_input(self):
        model = Emeans(1, 2, niter=1, batch_size=2, alpha0=0.5)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model.train(torch.full((2, 1), value))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_local_updates_do_not_synchronize(self):
        model = Emeans(1, 2, niter=1, batch_size=2, alpha0=0.5)
        model.begin(4, torch.tensor([[0.0], [10.0]], device="cuda"))
        rows = torch.tensor([[0.0], [1.0]], device="cuda")
        donor = torch.zeros((2, 1), device="cuda")
        child = donor.clone()
        model.update(model.accumulate(rows))  # Preparation outside sync-debug.
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            model.update(model.accumulate(rows))
            self.assertEqual(model.current_centroids().device, rows.device)
            # Split-separation has no standalone public operation; this
            # bounded private probe checks its no-host-sync repair path.
            e_means._separate_splits(donor, child)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        model.end_epoch()
        torch.testing.assert_close(child, torch.nextafter(donor, torch.full_like(donor, float("inf"))),
                                   rtol=0, atol=0)
        torch.testing.assert_close(model.finalize(), torch.tensor([[0.5], [10.0]], device="cuda"),
                                   rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
    def test_noncurrent_device_and_mismatch_recovery(self):
        current = torch.cuda.current_device()
        device = torch.device("cuda", (current+1) % torch.cuda.device_count())
        rows = torch.tensor([[0.0], [2.0], [10.0], [12.0]], device=device)
        initial = rows[[0, 2]].clone()
        model = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)
        model.begin(4, initial)
        self.assertEqual(model.current_centroids().device, device)
        self.assertFalse(model.current_centroids().requires_grad)
        with self.assertRaises(ValueError):
            model.accumulate(rows.to(torch.device("cuda", current)))
        model.update(model.accumulate(rows))
        model.end_epoch()
        result = model.finalize()
        self.assertEqual(result.device, device)
        self.assertEqual(torch.cuda.current_device(), current)
        torch.testing.assert_close(result, torch.tensor([[1.0], [11.0]], device=device), rtol=0, atol=0)

    def test_large_cpu_batch_preserves_sum_and_count(self):
        n = 2**24 + 2
        x = torch.full((n, 1), 0.25)
        km = Emeans(1, 1, niter=1, batch_size=n, alpha0=1.0)
        km.begin(n, torch.zeros((1, 1)))
        partial = km.accumulate(x[:2**24])
        self.assertEqual(partial["n"], 2**24)
        self.assertIsNone(partial["sums"])
        self.assertIsNone(partial["counts"])
        stats = km.accumulate(x[2**24:])
        self.assertEqual(stats["sums"].dtype, torch.float32)
        self.assertEqual(stats["counts"].dtype, torch.float32)
        self.assertEqual(stats["sums"][0, 0].item(), n * 0.25)
        self.assertEqual(stats["counts"][0].item(), n)
        km.update(stats)
        km.end_epoch()
        self.assertEqual(km.finalize()[0, 0].item(), 0.25)

    def test_caller_batches_match_matrix_training(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            for overrides in (
                dict(batch_size=4, alpha0=0.2,
                     assignment_chunk_size=2, centroid_chunk_size=2),
                {},
            ):
                with self.subTest(device=device, overrides=overrides):
                    # Binary values keep this exact fixture's FP32 sums independent
                    # of CUDA atomic addition order.
                    x = torch.arange(51, device=device, dtype=torch.float32)
                    x = x.reshape(17, 3) / 8
                    init = x[[0, 5, 10, 16]].clone()
                    args = dict(niter=3, seed=7, **overrides)
                    matrix, caller = Emeans(3, 4, **args), Emeans(3, 4, **args)
                    matrix.train(x, init_centroids=init)
                    caller.begin(len(x), init)
                    batch_size = caller.recipe["batch_size"]
                    for epoch in range(caller.niter):
                        generator = torch.Generator(device=device).manual_seed(
                            _epoch_seed(caller.seed, epoch)
                        )
                        order = torch.randperm(
                            len(x), generator=generator, device=device
                        )
                        for begin in range(0, len(x), batch_size):
                            stats = caller.accumulate(x[order[begin:begin + batch_size]])
                            caller.update(stats)
                        caller.end_epoch()
                    torch.testing.assert_close(
                        caller.finalize(), matrix.centroids, rtol=0, atol=0
                    )
                    torch.testing.assert_close(caller.obj, matrix.obj, rtol=0, atol=0)
                    self.assertEqual(caller.n_splits, matrix.n_splits)

    def test_closing_mean_and_assign(self):
        x = torch.tensor(
            [[0.0], [2.0], [10.0], [12.0]], requires_grad=True
        )
        init = torch.tensor([[0.0], [10.0]], requires_grad=True)
        km = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)

        centroids, captured_labels = km.train(x, init_centroids=init)

        torch.testing.assert_close(km.centroids, torch.tensor([[1.0], [11.0]]))
        self.assertEqual(km.obj[-1], 8.0)
        self.assertIs(centroids, km.centroids)
        self.assertIs(captured_labels, km.assignments)
        self.assertFalse(km.centroids.requires_grad)
        km.centroids.requires_grad_()
        distance, labels = km.assign(x)
        torch.testing.assert_close(distance, torch.ones(4))
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1]))
        self.assertFalse(distance.requires_grad)

    def test_alpha_one_preserves_an_empty_cluster(self):
        x = torch.tensor([[99.0], [0.0], [1.0], [100.0]])
        init = torch.tensor([[0.0], [100.0]])
        km = Emeans(1, 2, niter=1, batch_size=2, alpha0=1.0)

        km.train(x, init_centroids=init)

        torch.testing.assert_close(
            km.centroids, torch.tensor([[0.5], [99.5]])
        )

    def test_closing_mean_uses_only_the_final_pass(self):
        x = torch.tensor([[3.0], [4.0], [4.0], [6.0], [12.0]])
        init = torch.tensor([[0.0], [10.0]])
        km = Emeans(1, 2, niter=2, batch_size=5, alpha0=1.0)

        km.train(x, init_centroids=init)

        torch.testing.assert_close(
            km.centroids, torch.tensor([[4.25], [12.0]])
        )

    def test_blocked_assignment_matches_cdist(self):
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(23, 7, generator=generator)
        centroids = torch.randn(11, 7, generator=generator)
        km = Emeans(7, 11, niter=1, batch_size=11, alpha0=0.5,
                    centroid_chunk_size=3)
        km.train(centroids, init_centroids=centroids)
        self.assertEqual(km.recipe["centroid_chunk_size"], 3)

        distance, labels = km.assign(x)
        reference = torch.cdist(x, centroids).square()
        expected_distance, expected_labels = reference.min(dim=1)

        torch.testing.assert_close(
            distance, expected_distance, atol=2e-5, rtol=1e-5
        )
        torch.testing.assert_close(labels, expected_labels)

    def test_blocked_assignment_breaks_ties_by_index(self):
        x = torch.zeros((1, 2))
        centroids = torch.ones((5, 2))
        km = Emeans(2, 5, niter=1, batch_size=5, alpha0=0.5,
                    centroid_chunk_size=2)
        km.train(centroids, init_centroids=centroids)

        _, labels = km.assign(x)

        self.assertEqual(labels.item(), 0)

    def test_revival_occurs_during_training(self):
        x = torch.zeros((10, 1))
        init = torch.tensor([[0.0], [10.0]])
        km = Emeans(1, 2, niter=5, batch_size=10, alpha0=0.5)

        km.train(x, init_centroids=init)

        self.assertEqual(km.n_splits, 1)

    def test_chunking_does_not_change_updates(self):
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(41, 5, generator=generator)
        init = x[:4].clone()
        args = {"niter": 3, "batch_size": 13, "alpha0": 0.2, "seed": 7}
        a = Emeans(
            5,
            4,
            assignment_chunk_size=4,
            centroid_chunk_size=2,
            **args,
        )
        b = Emeans(
            5,
            4,
            assignment_chunk_size=64,
            centroid_chunk_size=8,
            **args,
        )

        a.train(x, init_centroids=init)
        b.train(x, init_centroids=init)

        torch.testing.assert_close(
            a.centroids, b.centroids, atol=2e-6, rtol=1e-6
        )
        torch.testing.assert_close(a.obj, b.obj, atol=2e-5, rtol=1e-6)

    def test_training_does_not_advance_global_rng(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            with self.subTest(device=device):
                x = torch.arange(80, dtype=torch.float32, device=device).reshape(20, 4)
                cpu_state = torch.random.get_rng_state()
                state = torch.cuda.get_rng_state() if device == "cuda" else cpu_state
                km = Emeans(4, 3, niter=1, batch_size=7, alpha0=0.2)
                km.train(x)
                after = torch.cuda.get_rng_state() if device == "cuda" else torch.random.get_rng_state()
                self.assertTrue(torch.equal(state, after))
                self.assertTrue(torch.equal(cpu_state, torch.random.get_rng_state()))

    def test_input_types_match(self):
        x = torch.randn(20, 4)
        init = x[:3].clone()
        km = Emeans(4, 3, niter=1, batch_size=7, alpha0=0.2)

        with self.assertRaises(TypeError):
            km.train(x.numpy(), init_centroids=init)
        with self.assertRaises(TypeError):
            km.train(x, init_centroids=init.numpy())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_replay_failure_drains_and_model_recovers(self):
        # This roster has a ragged tail under the existing three-epoch recipe;
        # the old 17-row case resolved to batch1 and never tested a tail.
        x = (torch.arange(93, device="cuda", dtype=torch.float32) / 8).reshape(31, 3)
        initial = x[[0, 5, 10, 16]].clone()
        saved_input, saved_initial = x.clone(), initial.clone()
        model = Emeans(3, 4, niter=3, seed=7, capture_assignments=True)
        replay, reset = torch.cuda.CUDAGraph.replay, torch.cuda.CUDAGraph.reset

        class InjectedFailure(BaseException):
            pass

        for error_type in (RuntimeError, InjectedFailure):
            with self.subTest(error_type=error_type.__name__):
                events, retired = [], []

                def fail_after_replay(
                    graph, _events=events, _error_type=error_type
                ):
                    # Torch's test-only delay is needed to prove work remains
                    # queued behind the delay at the host raise.
                    torch.cuda._sleep(50_000_000)
                    replay(graph)
                    done = torch.cuda.Event()
                    done.record()
                    _events.append(done)
                    self.assertFalse(done.query())
                    raise _error_type("injected replay failure")

                def checked_reset(graph, _events=events, _retired=retired):
                    if _events:
                        complete = _events[-1].query()
                        _retired.append(complete)
                        self.assertTrue(complete)  # BEFORE pool retirement.
                    reset(graph)

                with mock.patch.object(torch.cuda.CUDAGraph, "replay", fail_after_replay), \
                     mock.patch.object(torch.cuda.CUDAGraph, "reset", checked_reset):
                    with self.assertRaisesRegex(error_type, "injected replay failure"):
                        model.train(x, initial)
                self.assertEqual(len(events), 1)  # No retry.
                self.assertTrue(retired)
                self.assertIsNone(model.assignments)
                model.reset()
                recovered = []

                def counted_replay(graph):
                    recovered.append(True)  # noqa: B023 - rebound for closing replay
                    replay(graph)

                with mock.patch.object(torch.cuda.CUDAGraph, "replay", counted_replay):
                    model.train(x, initial)
                self.assertTrue(recovered)
                reference = Emeans(3, 4, niter=3, seed=7, assignment_chunk_size=2,
                                   capture_assignments=True)
                # The explicit override keeps this reference eager.
                reference.train(x, initial)
                torch.testing.assert_close(model.centroids, reference.centroids, rtol=0, atol=0)
                torch.testing.assert_close(model.obj, reference.obj, rtol=0, atol=0)
                torch.testing.assert_close(model.assignments, reference.assignments, rtol=0, atol=0)
                self.assertEqual(model.n_splits, reference.n_splits)
                torch.testing.assert_close(x, saved_input, rtol=0, atol=0)
                torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)

        # A one-epoch capture already owns closing sums at capture time.
        recovered = []
        single = Emeans(3, 4, niter=1, seed=7, capture_assignments=True)
        with mock.patch.object(torch.cuda.CUDAGraph, "replay", counted_replay):
            single.train(x, initial)
        self.assertTrue(recovered)
        eager = Emeans(3, 4, niter=1, seed=7, assignment_chunk_size=2,
                       capture_assignments=True)
        eager.train(x, initial)
        torch.testing.assert_close(single.centroids, eager.centroids, rtol=0, atol=0)
        torch.testing.assert_close(single.obj, eager.obj, rtol=0, atol=0)
        torch.testing.assert_close(single.assignments, eager.assignments, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
