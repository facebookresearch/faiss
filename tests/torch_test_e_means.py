# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import tempfile
import datetime
from unittest import mock

import torch
import numpy as np

from faiss.contrib.e_means import Emeans, DistributedEmeans
from faiss.contrib.e_means.e_means import _epoch_seed
from faiss.contrib.e_means.e_means_quantizer import CastQuantizer
from faiss.contrib.e_means import e_means
from e_means_test_utils import EmeansReferenceContracts


def _cpu_gloo_replica(rank, rendezvous):
    import torch.distributed as dist
    from faiss.contrib.e_means.e_means_distributed import _combine

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))
    try:
        check = unittest.TestCase()
        for numpy_state in (True, False):
            def array(values, _numpy_state=numpy_state):
                return (
                    np.asarray(values, dtype="float32")
                    if _numpy_state
                    else torch.tensor(values, dtype=torch.float32)
                )

            def host(value, _numpy_state=numpy_state):
                return value if _numpy_state else value.detach().numpy()
            for transport in ("dense", "sparse", "auto"):
                initial = array([[0.], [2.], [100.], [200.]])
                caller_initial = initial + rank * 1000
                saved_initial = host(caller_initial).copy()
                rows = array(np.array([0., .25, 2., 2.25])[np.arange(10) % 4, None])
                args = dict(niter=2, batch_size=4, alpha0=0.5, seed=7,
                            capture_assignments=True)
                model = DistributedEmeans(1, 4, group=dist.group.WORLD, transport=transport, **args)
                reference = Emeans(1, 4, **args)
                with check.assertRaisesRegex(ValueError, "divisible"):
                    model.begin(9, caller_initial)
                with check.assertRaises(RuntimeError):
                    model.accumulate(rows[:1])
                with check.assertRaises(NotImplementedError):
                    model.train(rows)
                model.begin(len(rows), caller_initial)
                reference.begin(len(rows), initial)
                np.testing.assert_array_equal(host(model.current_centroids()), host(initial))
                np.testing.assert_array_equal(host(caller_initial), saved_initial)
                if rank:
                    model._split_rng.seed(123456) if numpy_state else model._split_rng.manual_seed(123456)
                for epoch in range(2):
                    for begin in range(0, len(rows), 4):
                        n = min(4, len(rows) - begin)
                        start = begin + rank * (n // 2)
                        local = rows[start:start + n // 2]
                        if epoch == begin == 0:
                            # External LOCAL statistics without calling accumulate:
                            # Initial labels and distances have exact dyadic truth.
                            labels = (host(local)[:, 0] >= 2).astype("int64")
                            sums = array(np.bincount(labels, weights=host(local)[:, 0], minlength=4)[:, None])
                            counts = array(np.bincount(labels, minlength=4))
                            objective = np.float64(((host(local)[:, 0] - 2 * labels) ** 2).sum())
                            if numpy_state:
                                sums = sums[::-1].copy()[::-1]  # Valid negative stride.
                                sums.flags.writeable = counts.flags.writeable = False
                            else:
                                objective = torch.tensor(objective, dtype=torch.float64)
                            stats = dict(sums=sums, counts=counts, objective=objective, n=len(local))
                        else:
                            before = host(model.current_centroids()).copy()
                            if rank == 0 and len(local) > 1:
                                partial = model.accumulate(local[:1])
                                assert partial["n"] == 1 and partial["sums"] is None
                                assert partial["counts"] is None
                                with check.assertRaisesRegex(ValueError, "complete the pending"):
                                    model.update({})
                                np.testing.assert_array_equal(host(model.current_centroids()), before)
                                stats = model.accumulate(local[1:])
                            else:
                                stats = model.accumulate(local)
                            if epoch == 1:
                                supplied = local[1:] if rank == 0 and len(local) > 1 else local
                                labels = ((host(supplied)[:, None, :] - before[None, :, :]) ** 2).sum(2).argmin(1)
                                np.testing.assert_array_equal(host(stats["labels"]), labels)
                            stats = {name: value.copy() if isinstance(value, np.ndarray)
                                     else value.clone() if isinstance(value, torch.Tensor) else value
                                     for name, value in stats.items()}
                        saved = {name: host(stats[name]).copy() for name in ("sums", "counts")}
                        with check.assertRaisesRegex(ValueError, "prescribed batch size"):
                            model.update(dict(stats, n=n))
                        model.update(stats)
                        assert model._epoch_rows == begin + n
                        for name, value in saved.items():
                            np.testing.assert_array_equal(host(stats[name]), value)
                        reference.update(reference.accumulate(rows[begin:begin + n]))
                        np.testing.assert_allclose(host(model.current_centroids()), host(reference.current_centroids()), rtol=1e-6, atol=1e-6)
                        np.testing.assert_array_equal(host(model._pass_counts), host(reference._pass_counts))
                        if model._pass_sums is not None:
                            np.testing.assert_allclose(host(model._pass_sums), host(reference._pass_sums), rtol=1e-6, atol=1e-6)
                    model.end_epoch()
                    reference.end_epoch()
                    np.testing.assert_allclose(float(model._objective), float(reference._objective), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(host(model.finalize()), host(reference.finalize()), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(host(model.obj), host(reference.obj), rtol=1e-6, atol=1e-6)
                assert model.assignments is None  # Caller chunks own labels, not a dataset result.
                assert model.n_splits == reference.n_splits and model.n_splits >= 2
                final = model.centroids
                saved_final = host(final).copy()
                model.reset()
                np.testing.assert_array_equal(host(final), saved_final)
            # Fail after real collective execution, then recover only via begin.
            class CollectiveFailure(BaseException):
                pass
            model = DistributedEmeans(1, 1, group=dist.group.WORLD,
                                      transport="dense", niter=1,
                                      batch_size=2, alpha0=0.5)
            initial = array([[0.]])
            broadcast = dist.broadcast
            def failed_broadcast(*args, _broadcast=broadcast, **kwargs):
                _broadcast(*args, **kwargs)
                raise CollectiveFailure()
            with mock.patch.object(dist, "broadcast", side_effect=failed_broadcast):
                with check.assertRaises(CollectiveFailure):
                    model.begin(2, initial)
            with check.assertRaises(RuntimeError):
                model.current_centroids()
            model.begin(2, initial)
            stats = model.accumulate(array([[.25]]))
            with check.assertRaisesRegex(ValueError, "prescribed batch size"):
                model.update(dict(stats, n=2))
            np.testing.assert_array_equal(host(model.current_centroids()), host(initial))
            def failed_combine(*args, **kwargs):
                _combine(*args, **kwargs)
                raise CollectiveFailure()
            with mock.patch("faiss.contrib.e_means.e_means_distributed._combine", side_effect=failed_combine):
                with check.assertRaises(CollectiveFailure):
                    model.update(stats)
            for call in (
                lambda _model=model: _model.accumulate(array([[.25]])),
                lambda _model=model, _stats=stats: _model.update(_stats),
                model.end_epoch,
                model.finalize,
                model.current_centroids,
            ):
                with check.assertRaises(RuntimeError):
                    call()
            model.begin(2, initial)
            model.update(model.accumulate(array([[.25]])))
            model.end_epoch()
            np.testing.assert_array_equal(host(model.finalize()), [[.25]])
            if numpy_state:
                # A reversed singleton can report C-contiguous yet retain a
                # negative stride. Exercise a real one-donor broadcast too.
                args = dict(niter=1, batch_size=2, alpha0=0.5, seed=7)
                model = DistributedEmeans(1, 2, group=dist.group.WORLD,
                                          transport="dense", **args)
                reference = Emeans(1, 2, **args)
                initial = array([[0.], [100.]])
                model.begin(10, initial + rank * 1000)
                reference.begin(10, initial)
                for _ in range(5):
                    model.update(model.accumulate(array([[.25]])))
                    reference.update(reference.accumulate(array([[.25], [.25]])))
                assert model.n_splits == reference.n_splits == 1
                model.end_epoch()
                reference.end_epoch()
                np.testing.assert_allclose(model.finalize(), reference.finalize(), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(model.obj, reference.obj, rtol=1e-6, atol=1e-6)
            # Packing edge cases independent of row-count validation: a rank
            # with no occupied IDs, and then all ranks with no occupied IDs.
            for all_empty in (False, True):
                counts = array([0., 0.] if rank or all_empty else [2., 0.])
                sums = array([[0.], [0.]])
                workspace = np.empty((2, 2), dtype="float32") if numpy_state else torch.empty((2, 2))
                ids, combined_sums, combined_counts = _combine(sums, counts, workspace, dist.group.WORLD, "sparse", 4)
                assert isinstance(ids, np.ndarray if numpy_state else torch.Tensor)
                assert isinstance(combined_sums, np.ndarray if numpy_state else torch.Tensor)
                np.testing.assert_array_equal(host(ids), [] if all_empty else [0])
                np.testing.assert_array_equal(host(combined_counts), [] if all_empty else [2.])
                assert combined_sums.shape == (0 if all_empty else 1, 1)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestTorchEmeans(EmeansReferenceContracts, unittest.TestCase):
    @unittest.skipUnless(torch.distributed.is_available() and
                         torch.distributed.is_gloo_available(), "Gloo unavailable")
    def test_cpu_gloo_caller_owned_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(
                _cpu_gloo_replica, args=("file://" + directory + "/group",),
                nprocs=2, join=True, start_method="spawn"
            )

    def test_stateless_cast_and_cpu_scope(self):
        for bad_dimension, bad_dtype in ((0, torch.float16), (1, torch.float32)):
            with self.assertRaises(ValueError):
                CastQuantizer(bad_dimension, bad_dtype)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q = CastQuantizer(2, dtype)
                q.train()
                self.assertTrue(q.is_trained)
                self.assertEqual(q.code_size, 4)
                self.assertIs(q.to("cpu"), q)
                x = torch.tensor([[1.001, 9., 3.001, 9.],
                                  [2.001, 9., 4.001, 9.]], requires_grad=True)[:, ::2]
                saved = x.detach().clone()
                codes = q.compute_codes(x)
                self.assertTrue(codes.is_contiguous())
                self.assertFalse(codes.requires_grad)
                torch.testing.assert_close(q.decode(codes), x.to(dtype).float(), rtol=0, atol=0)
                torch.testing.assert_close(x, saved, rtol=0, atol=0)
                q.validate(x)
                for bad in (x[0], x[:, :1], x.detach().to_sparse()):
                    with self.assertRaises(ValueError):
                        q.compute_codes(bad)
                with self.assertRaises(TypeError):
                    q.compute_codes(x.to(torch.int32))
                with self.assertRaises(TypeError):
                    q.decode(x)
                with self.assertRaises(ValueError):
                    q.to("meta")
                for value in (float("nan"), float("inf")):
                    with self.assertRaises(ValueError):
                        q.validate(torch.full((1, 2), value))
                for rows in (x, saved.numpy()):
                    with self.assertRaisesRegex(ValueError, "quantizer=None"):
                        Emeans(2, 1, quantizer=q).train(rows)
                with self.assertRaises(ValueError):
                    Emeans(3, 1, quantizer=q)
                # Native CPU low-precision acceptance begins at P03.
                model = Emeans(2, 1, niter=1, batch_size=2, alpha0=0.5)
                model.train(codes)
                torch.testing.assert_close(model.centroids, codes.float().mean(0, keepdim=True))
                self.assertEqual(model.centroids.dtype, torch.float32)
                model.begin(2, codes[:1].float())
                partial = model.accumulate(codes[:1])
                self.assertIsNone(partial["sums"])
                self.assertIsNone(partial["counts"])
                with self.assertRaisesRegex(ValueError, "complete the pending"):
                    model.update({})
                stats = model.accumulate(codes[1:])
                model.update(stats)
                model.end_epoch()
                torch.testing.assert_close(model.finalize(), codes.float().mean(0, keepdim=True))
        half = CastQuantizer(1, torch.float16)
        half.validate(torch.tensor([[65504.]]))
        with self.assertRaises(ValueError):
            half.validate(torch.tensor([[65505.]]))
        # Unchecked conversion follows Torch; it is not a validating API.
        torch.testing.assert_close(half.compute_codes(torch.tensor([[65520.]])),
                                   torch.tensor([[65520.]]).half(), rtol=0, atol=0)

    def _known_centroids(self, centroids, **kwargs):
        """Install fixed means through the documented external-statistics API."""
        k, d = centroids.shape
        model = Emeans(d, k, niter=1, batch_size=k, alpha0=1, **kwargs)
        model.begin(k, centroids)
        model.update(dict(sums=centroids.clone(),
                          counts=torch.ones(k, device=centroids.device),
                          objective=torch.zeros((), dtype=torch.float64, device=centroids.device),
                          n=k))
        model.end_epoch()
        torch.testing.assert_close(model.finalize(), centroids, rtol=0, atol=0)
        return model

    @unittest.skipUnless(torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
                         and torch.cuda.get_device_capability() >= (8, 0), "NVIDIA SM80 required")
    def test_represented_assignment_norms_ties_and_high_finite(self):
        for dtype in (torch.float16, torch.bfloat16):
            for n, k, d in ((17, 7, 3), (31, 13, 65)):
                with self.subTest(dtype=dtype, shape=(n, k, d)):
                    rng = torch.Generator(device="cuda").manual_seed(123+d)
                    rows = torch.randn(n, d, generator=rng, device="cuda")/4
                    centroids = torch.randn(k, d, generator=rng, device="cuda")/4
                    reference = (
                        rows.to(dtype).float()[:, None]
                        - centroids.to(dtype).float()[None]
                    ).square().sum(2)
                    expected_distance, expected_labels = reference.min(1)
                    for quantizer in (None, CastQuantizer(d, dtype)):
                        model = self._known_centroids(
                            centroids,
                            quantizer=quantizer,
                            assignment_chunk_size=5,
                            centroid_chunk_size=2,
                        )
                        distance, labels = model.assign(rows.to(dtype) if quantizer is None else rows)
                        torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
                        torch.testing.assert_close(distance, expected_distance, atol=2e-4, rtol=2e-5)
                        self.assertEqual((distance.dtype, labels.dtype), (torch.float32, torch.int64))
            small = (1.0001, 1.0004) if dtype == torch.float16 else (1.001, 1.002)
            centroids = torch.tensor([[small[0], 0.], [small[1], 0.], [3., 0.]], device="cuda")
            rows = centroids[1:2].clone()
            self.assertEqual(int(torch.cdist(rows, centroids).argmin(1)), 1)
            torch.testing.assert_close(centroids[0].to(dtype), centroids[1].to(dtype), rtol=0, atol=0)
            model = self._known_centroids(centroids, centroid_chunk_size=1)
            distance, labels = model.assign(rows.to(dtype))
            self.assertEqual(int(labels[0]), 0)
            self.assertEqual(float(distance[0]), 0)
        rows = torch.tensor([[-0.4678138, 0.03789843, -0.6960376]], device="cuda")
        centroids = torch.tensor([[-0.8843690, 1.0697185, -0.7981653],
                                  [0.4984003, 0.5734861, -0.5365824]], device="cuda")
        represented = (
            rows.bfloat16().float()[:, None]
            - centroids.bfloat16().float()[None]
        ).square().sum(2)
        wrong_norm = centroids.square().sum(1) - 2 * (
            rows.bfloat16().float()[:, None]
            * centroids.bfloat16().float()[None]
        ).sum(2)
        self.assertEqual(int(represented.argmin(1)), 0)
        self.assertEqual(int(wrong_norm.argmin(1)), 1)
        model = self._known_centroids(centroids, centroid_chunk_size=1)
        distance, labels = model.assign(rows.bfloat16())
        self.assertEqual(int(labels[0]), 0)
        torch.testing.assert_close(distance, represented[:, 0], atol=1e-6, rtol=0)
        centroids = torch.tensor([[300.], [302.]], device="cuda")
        rows = torch.tensor([[300.], [301.]], device="cuda", dtype=torch.float16)
        self.assertGreater(float(centroids[0].square().sum()), torch.finfo(torch.float16).max)
        distance, labels = self._known_centroids(centroids, centroid_chunk_size=1).assign(rows)
        torch.testing.assert_close(labels, torch.zeros(2, dtype=torch.int64, device="cuda"), rtol=0, atol=0)
        torch.testing.assert_close(distance, torch.tensor([0., 1.], device="cuda"), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
                         and torch.cuda.get_device_capability() >= (8, 0), "NVIDIA SM80 required")
    def test_represented_empty_metadata_split_and_range(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = Emeans(1, 2, niter=1, batch_size=2, alpha0=0.5,
                               capture_assignments=True)
                model.begin(4, torch.tensor([[0.], [10.]], device="cuda"))
                empty = model.accumulate(torch.empty((0, 1), dtype=dtype, device="cuda"))
                self.assertEqual(empty["n"], 0)
                with self.assertRaisesRegex(ValueError, "dtype must be consistent"):
                    model.accumulate(torch.empty((0, 1), device="cuda"))
                model.update(dict(sums=torch.tensor([[2.], [0.]], device="cuda"),
                                  counts=torch.tensor([2., 0.], device="cuda"),
                                  objective=torch.zeros((), dtype=torch.float64, device="cuda"), n=2))
                rows = torch.ones((2, 1), dtype=dtype, device="cuda")
                before = model.current_centroids().clone()
                partial = model.accumulate(rows[:1])
                self.assertEqual(partial["n"], 1)
                self.assertIsNone(partial["sums"])
                self.assertIsNone(partial["counts"])
                with self.assertRaisesRegex(ValueError, "dtype must be consistent"):
                    model.accumulate(rows[1:].float())
                torch.testing.assert_close(model.current_centroids(), before, rtol=0, atol=0)
                stats = model.accumulate(rows[1:])
                torch.testing.assert_close(stats["labels"], torch.zeros(1, dtype=torch.int64, device="cuda"), rtol=0, atol=0)
                self.assertEqual(stats["sums"].dtype, torch.float32)
                self.assertEqual(stats["counts"].dtype, torch.float32)
                model.update(stats)
                model.end_epoch()
                torch.testing.assert_close(model.finalize(), torch.tensor([[1.], [10.]], device="cuda"), rtol=0, atol=0)
                maximum = torch.finfo(dtype).max
                donor = torch.tensor([[0., 0.], [1., 1.], [1., 2.], [maximum, 0.]], device="cuda")
                child = torch.tensor([[0., 0.], [1.0001, 0.9999], [2., 1.], [maximum, 0.]], device="cuda")
                ordinary = child[2].clone()
                previous = torch.cuda.get_sync_debug_mode()
                try:
                    torch.cuda.set_sync_debug_mode("error")
                    e_means._separate_splits(donor, child, dtype)
                finally:
                    torch.cuda.set_sync_debug_mode(previous)
                self.assertFalse(bool((donor.to(dtype) == child.to(dtype)).all(1).any()))
                torch.testing.assert_close(child[2], ordinary, rtol=0, atol=0)
                self.assertLess(float(child[3, 0]), maximum)
                torch.testing.assert_close(child[:2, 1], torch.tensor([0., 0.9999], device="cuda"), rtol=0, atol=0)
        for dtype, value in ((torch.float16, 65504.), (torch.bfloat16, 70000.)):
            rows = torch.full((2, 1), value, device="cuda", dtype=dtype)
            model = Emeans(1, 1, niter=1, batch_size=2, alpha0=0.5)
            model.train(rows, rows[:1].float())
            torch.testing.assert_close(model.centroids, rows[:1].float(), rtol=0, atol=0)
            torch.testing.assert_close(model.assign(rows)[0], torch.zeros(2, device="cuda"), rtol=0, atol=0)
        model = Emeans(1, 2, niter=1, batch_size=2, alpha0=0.5)
        rows = torch.full((10, 1), 65504., device="cuda", dtype=torch.float16)
        before_split = []
        apply_splits = model._apply_splits

        def checked_split(*args):
            centroids, usage = model._centroids.clone(), model._usage.clone()
            try:
                return apply_splits(*args)
            except ValueError:
                before_split.append(True)
                torch.testing.assert_close(model._centroids, centroids, rtol=0, atol=0)
                torch.testing.assert_close(model._usage, usage, rtol=0, atol=0)
                raise

        with mock.patch.object(model, "_apply_splits", checked_split), self.assertRaisesRegex(ValueError, "65504"):
            model.train(rows, rows[:2].float())
        self.assertTrue(before_split)
        self.assertEqual(model.n_splits, 0)
        rows = torch.zeros((2, 2), device="cuda", dtype=torch.bfloat16)
        with mock.patch.object(torch.version, "hip", "test"), self.assertRaisesRegex(RuntimeError, "SM80"):
            Emeans(2, 2).train(rows)
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)), self.assertRaisesRegex(RuntimeError, "SM80"):
            Emeans(2, 2).train(rows)

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
        and torch.cuda.get_device_capability() >= (8, 0), "NVIDIA SM80 required"
    )
    def test_native_and_cast_represented_means(self):
        x = torch.tensor([[1.001], [1.002], [3.001], [3.002]], device="cuda")
        mean_initial = torch.tensor([[1.0], [3.0]], device="cuda")
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q = CastQuantizer(1, dtype)
                codes = q.compute_codes(x)
                native = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5)
                cast = Emeans(1, 2, niter=1, batch_size=4, alpha0=0.5, quantizer=q)
                native.train(codes, mean_initial)
                cast.train(x, mean_initial)
                expected = codes.float().reshape(2, 2, 1).mean(dim=1)
                torch.testing.assert_close(native.centroids, expected, atol=0, rtol=0)
                torch.testing.assert_close(cast.centroids, expected, atol=0, rtol=0)
                torch.testing.assert_close(native.obj, cast.obj, atol=0, rtol=0)
                dn, ln = native.assign(codes)
                dc, lc = cast.assign(x)
                torch.testing.assert_close(dn, dc, atol=0, rtol=0)
                torch.testing.assert_close(ln, lc)
                self.assertEqual((native.centroids.dtype, dn.dtype),
                                 (torch.float32, torch.float32))
                # Retained decimal T2/N17/B4 fixture includes the final row.
                rows = torch.arange(34, device="cuda", dtype=torch.float32).reshape(17, 2)/10
                represented = CastQuantizer(2, dtype).compute_codes(rows)
                initial = represented[[0, 5, 10, 16]].float()
                saved_rows, saved_initial = rows.clone(), initial.clone()
                args = dict(niter=2, batch_size=4, alpha0=0.5, seed=7)
                reference = Emeans(2, 4, assignment_chunk_size=2, centroid_chunk_size=2, **args)
                reference.train(represented, initial)
                q = CastQuantizer(2, dtype)
                cast = Emeans(2, 4, quantizer=q, assignment_chunk_size=2, centroid_chunk_size=2, **args)
                cast.train(rows, initial)
                unblocked = Emeans(2, 4, assignment_chunk_size=64, centroid_chunk_size=16, **args)
                unblocked.train(represented, initial)
                caller = Emeans(2, 4, quantizer=q, assignment_chunk_size=2, centroid_chunk_size=2, **args)
                caller_initial = initial.clone()
                caller.begin(len(rows), caller_initial)
                caller_initial.fill_(-7)
                current = caller.current_centroids()
                self.assertEqual((current.dtype, current.device), (torch.float32, initial.device))
                torch.testing.assert_close(current, initial, rtol=0, atol=0)
                for epoch in range(2):
                    generator = torch.Generator(device="cuda").manual_seed(_epoch_seed(7, epoch))
                    order = torch.randperm(len(rows), generator=generator, device="cuda")
                    for begin in range(0, len(rows), 4):
                        batch = rows[order[begin:begin+4]]
                        before = caller.current_centroids().clone()
                        for start in range(0, len(batch), 2):
                            stats = caller.accumulate(batch[start:start+2])
                            if start + 2 < len(batch):
                                self.assertIsNone(stats["sums"])
                                self.assertIsNone(stats["counts"])
                                torch.testing.assert_close(caller.current_centroids(), before, rtol=0, atol=0)
                        self.assertEqual(stats["sums"].dtype, torch.float32)
                        self.assertEqual(stats["counts"].dtype, torch.float32)
                        self.assertEqual(stats["objective"].dtype, torch.float64)
                        caller.update(stats)
                    caller.end_epoch()
                result = caller.finalize()
                for other in (cast, unblocked, caller):
                    torch.testing.assert_close(other.centroids, reference.centroids, atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(other.obj, reference.obj, atol=3e-5, rtol=1e-6)
                    for field in ("nsplit", "imbalance_factor"):
                        self.assertEqual([s[field] for s in other.iteration_stats],
                                         [s[field] for s in reference.iteration_stats])
                torch.testing.assert_close(cast.assign(rows)[0], reference.assign(represented)[0], atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(rows, saved_rows, rtol=0, atol=0)
                torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)
                saved_result = result.clone()
                caller.reset()
                caller.train(rows, initial)
                torch.testing.assert_close(caller.centroids, saved_result, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(result, saved_result, rtol=0, atol=0)

    def test_local_update_has_no_group_argument(self):
        model = Emeans(1, 1, niter=1, batch_size=2, alpha0=0.5)
        model.begin(4, torch.zeros((1, 1)))
        stats = model.accumulate(torch.zeros((2, 1)))
        with self.assertRaises(TypeError):
            model.update(stats, group=object())
        model.update(stats)
        stats = model.accumulate(torch.ones((2, 1)))
        with self.assertRaises(TypeError):
            model.update(stats, group=object())
        model.update(stats)
        model.end_epoch()
        torch.testing.assert_close(model.finalize(), torch.tensor([[0.5]]))


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
        cases = [(torch.float32, None)]
        if not torch.version.hip and torch.cuda.get_device_capability() >= (8, 0):
            cases.extend(((torch.bfloat16, None),
                          (torch.float32, CastQuantizer(3, torch.bfloat16))))
        for dtype, quantizer in cases:
            with self.subTest(dtype=dtype, quantizer=quantizer is not None):
                self._check_cuda_replay_failure(dtype, quantizer)

    def _check_cuda_replay_failure(self, dtype, quantizer):
        x = (torch.arange(93, device="cuda", dtype=torch.float32) / 8).reshape(31, 3)
        initial = x[[0, 5, 10, 16]].clone()
        x = x.to(dtype)
        saved_input, saved_initial = x.clone(), initial.clone()
        model = Emeans(3, 4, niter=3, seed=7, quantizer=quantizer, capture_assignments=True)
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
                                   quantizer=quantizer, capture_assignments=True)
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
        single = Emeans(3, 4, niter=1, seed=7, quantizer=quantizer,
                        capture_assignments=True)
        with mock.patch.object(torch.cuda.CUDAGraph, "replay", counted_replay):
            single.train(x, initial)
        self.assertTrue(recovered)
        eager = Emeans(3, 4, niter=1, seed=7, assignment_chunk_size=2,
                       quantizer=quantizer, capture_assignments=True)
        eager.train(x, initial)
        torch.testing.assert_close(single.centroids, eager.centroids, rtol=0, atol=0)
        torch.testing.assert_close(single.obj, eager.obj, rtol=0, atol=0)
        torch.testing.assert_close(single.assignments, eager.assignments, rtol=0, atol=0)

    def test_assign_resolves_independently_without_changing_training_recipe(self):
        model = Emeans(1, 2, niter=1, batch_size=4, alpha0=.5)
        rows = torch.tensor([[0.], [2.], [10.], [12.]])
        centroids, captured = model.train(rows, rows[[0, 2]])
        recipe = dict(model.recipe)
        model._input_dtype = torch.bfloat16  # Represents a completed BF16 run.
        distance, labels = model.assign(rows)
        torch.testing.assert_close(distance, torch.ones(4), rtol=0, atol=0)
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1]), rtol=0, atol=0)
        self.assertIs(model.centroids, centroids)
        self.assertIs(model.assignments, captured)
        self.assertEqual(model.recipe, recipe)
        self.assertEqual(model._input_dtype, torch.bfloat16)
        self.assertEqual(model.assignment_recipe['representation'], 'float32')
        self.assertEqual(model.assignment_recipe['assignment_backend'], 'native')


if __name__ == "__main__":
    unittest.main()
