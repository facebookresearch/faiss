# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Local IVF lifecycle, shared engines, and CPU/Gloo ownership integration."""

import datetime
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from faiss.contrib.e_means import EmeansIVF, DistributedEmeansIVF, EmeansShardedCentroidsIVF
from faiss.contrib.e_means.ivf_assignment import IVFAssignment, TorchIVFEngine, FaissIVFEngine
from faiss.contrib.e_means.e_means_distributed import _fill_initial_axes


def _data():
    generator = torch.Generator().manual_seed(81)
    return torch.randn(160, 8, generator=generator), torch.randn(8, 8, generator=generator)


def _options(engine="torch", **kwargs):
    return dict(nlist=4, nprobe=4, coarse_niter=2, nepochs=2.6,
                batch_size=32, alpha0=.4, assignment_chunk_size=11,
                ivf_engine=engine, ivf_tie_policy="lowest_id" if engine == "torch" else "faiss", **kwargs)


def _stream(model, rows, n, initial):
    model.begin(n, initial)
    for budget in model.recipe["epoch_rows"]:
        share = budget // getattr(model, "_group_size", 1)
        for begin in range(0, share, model.recipe["batch_size"] // getattr(model, "_group_size", 1)):
            batch = rows[begin:min(share, begin + model.recipe["batch_size"] // getattr(model, "_group_size", 1))]
            # Partial accumulation must preserve one frozen logical index.
            model.accumulate(batch[:1])
            model.update(model.accumulate(batch[1:]))
        model.end_epoch()
    return model.finalize()


def _replica_axes(model):
    # The replicated base uses a common CPU PCG64 stream. Give the local
    # reference identical directions when comparing the optimization steps.
    axes = torch.empty_like(model._centroids)
    _fill_initial_axes(axes, np.random.Generator(np.random.PCG64(model.seed)))
    return axes


def _gloo_worker(rank, ranks, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=ranks,
                            timeout=datetime.timedelta(seconds=90))
    try:
        check = unittest.TestCase()
        rows, initial = _data()
        # Arrange each batch's rank shares so the concatenated contributions
        # match the local reference's order (sparse reduction may reorder).
        local = rows.reshape(-1, ranks, 32 // ranks, 8)[:, rank].reshape(-1, 8)
        for engine in ("torch", "faiss"):
            for transport in ("dense", "sparse"):
                reference = EmeansIVF(8, 8, **_options(engine))
                reference._initial_removal_axes = lambda: _replica_axes(reference)
                expected = _stream(reference, rows, len(rows), initial)
                model = DistributedEmeansIVF(8, 8, group=dist.group.WORLD,
                                             transport=transport, **_options(engine))
                # Non-root initialization deliberately differs.
                result = _stream(model, local, len(rows), initial if rank == 0 else initial + 50)
                torch.testing.assert_close(result, expected, rtol=2e-4, atol=2e-4)
                check.assertEqual(model.n_splits, reference.n_splits)
                torch.testing.assert_close(model.obj, reference.obj, rtol=2e-5, atol=2e-5)
                peers = [torch.empty_like(result) for _ in range(ranks)]
                dist.all_gather(peers, result)
                for peer in peers:
                    torch.testing.assert_close(result, peer, rtol=0, atol=0)
            sharded = EmeansShardedCentroidsIVF(8, 8, group=dist.group.WORLD, **_options(engine))
            result = _stream(sharded, local, len(rows), initial[rank * 4:(rank + 1) * 4])
            check.assertEqual(result.shape, (4, 8))
            check.assertTrue(torch.isfinite(result).all())
            check.assertEqual(sharded.recipe["fine_dtype"], "fp32")
            resident = EmeansShardedCentroidsIVF(8, 8, group=dist.group.WORLD,
                **dict(_options(engine), nepochs=.6))
            centers, codes = resident.train(local, initial[rank * 4:(rank + 1) * 4])
            check.assertTrue(torch.isfinite(centers).all())
            check.assertEqual(int((codes >= 0).sum()), 48)

        # Force cross-owner cost-based splits and exercise empty all-to-all
        # senders. These are production begin/accumulate calls on CPU/Gloo.
        initial = torch.tensor([[0., 0.], [2., 0.], [2., 0.], [10., 0.]])
        model = EmeansShardedCentroidsIVF(2, 4, group=dist.group.WORLD,
            nlist=2, nprobe=2, coarse_niter=1, nepochs=2, batch_size=4)
        model.begin(40, initial[rank * 2:(rank + 1) * 2])
        model.accumulate(initial[rank * 2:(rank + 1) * 2])
        model._usage[:] = 1
        state = torch.tensor([[1., 0., 0.], [10., 0., 10.], [2., 0., 0.], [20., 0., 10.]])
        model._removal_stats[:] = state[rank * 2:(rank + 1) * 2]
        model._axes[:] = torch.tensor([1., 0.])
        check.assertEqual(model._revive_unused(model._split_rng, chunk_rows=1), 2)
        check.assertEqual(model.n_splits, 2)
        check.assertTrue(model._moved.all())
    finally:
        dist.destroy_process_group()


def _cuda_local_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    check = unittest.TestCase()
    rows, initial = (tensor.cuda() for tensor in _data())
    # A local GPU trainer must work before any process group exists.
    check.assertFalse(dist.is_initialized())
    local = EmeansIVF(8, 8, **_options())
    expected = _stream(local, rows, len(rows), initial)
    dist.init_process_group("nccl", init_method=rendezvous, rank=0, world_size=1,
                            timeout=datetime.timedelta(seconds=90))
    try:
        owner = EmeansShardedCentroidsIVF(8, 8, group=dist.group.WORLD, **_options())
        actual = _stream(owner, rows, len(rows), initial)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(owner.obj, local.obj, rtol=2e-5, atol=2e-5)
        check.assertEqual(owner.n_splits, local.n_splits)
        for backend in ("native", "triton"):
            model = EmeansIVF(8, 8, **dict(_options(), nepochs=.6, batch_size=8,
                                         assignment_backend=backend))
            centers, labels = model.train(rows, torch.zeros_like(initial))
            check.assertGreater(model.n_splits, 0)
            check.assertEqual(int((labels >= 0).sum()), 96)
            check.assertTrue(torch.isfinite(centers).all())
            distance, ids = model.assign(rows)
            torch.testing.assert_close(distance, (rows - centers[ids]).square().sum(1))
    finally:
        dist.destroy_process_group()


class TestIVFTraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_engines_shared_probes_and_rebuild(self):
        rows, candidates = _data()
        coarse = candidates[:4]
        torch_index = TorchIVFEngine(coarse, nprobe=4, query_chunk=7)
        faiss_index = FaissIVFEngine(coarse, nprobe=4, query_chunk=7)
        for centroids in (candidates, candidates + .3):
            torch_index.add(centroids)
            faiss_index.add(centroids)
            probes = torch_index.route(rows)
            expected, ids = torch_index.search_2nn(rows, coarse_idx=probes)
            actual, labels = faiss_index.search_2nn(rows, coarse_idx=probes)
            torch.testing.assert_close(labels, ids.long())
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        for index in (torch_index, faiss_index):
            scores, labels = index.search_2nn(rows[:0])
            self.assertEqual(scores.shape, (0, 2))
            index.add(candidates[:1])
            scores, labels = index.search_2nn(rows[:3])
            self.assertTrue((labels[:, 0] == 0).all())
            self.assertTrue((labels[:, 1] == -1).all())
            self.assertTrue(torch.isinf(scores[:, 1]).all())

    def test_explicit_faiss_tie_policy_and_constraints(self):
        with self.assertRaises(ValueError):
            EmeansIVF(8, 8, ivf_engine="faiss", nlist=4)
        for options in (dict(spherical=True), dict(nlist=9), dict(ivf_engine="unknown")):
            with self.assertRaises(ValueError):
                EmeansIVF(8, 8, **options)
        model = EmeansIVF(8, 8, **_options())
        rows, initial = _data()
        model.begin(len(rows), initial)
        with self.assertRaises(ValueError):
            model.accumulate(rows.half())

    def test_streaming_statistics_snapshot_and_query_isolation(self):
        rows, initial = _data()
        for engine in ("torch", "faiss"):
            model = EmeansIVF(8, 8, **_options(engine))
            model.begin(len(rows), initial)
            model.assign(rows[:3])  # Evaluation before the first training snapshot.
            self.assertIsNone(model._ivf)
            partial = model.accumulate(rows[:9])
            index = model._ivf
            self.assertIsNone(partial["sums"])
            with mock.patch.object(index, "add", side_effect=AssertionError("query mutated training index")):
                distance, labels = model.assign(rows[:3])
            self.assertIs(model._ivf, index)
            self.assertTrue(torch.isfinite(distance).all())
            with mock.patch.object(index, "add", side_effect=AssertionError("partial rebuilt index")):
                stats = model.accumulate(rows[9:32])
            all_distances = (rows[:32, None] - initial[None]).square().sum(-1)
            order = all_distances.argsort(dim=1, stable=True)
            expected = torch.zeros(8, 8).index_add_(0, order[:, 0], rows[:32])
            costs = all_distances.gather(1, order[:, 1:2]) - all_distances.gather(1, order[:, :1])
            removal = torch.zeros(8).index_add_(0, order[:, 0], costs[:, 0])
            torch.testing.assert_close(stats["sums"], expected)
            torch.testing.assert_close(stats["removal"][:, -1], removal, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(stats["objective"], all_distances[:, :].min(1).values.double().sum())
            model.update(stats)
            with mock.patch.object(index, "add", wraps=index.add) as rebuild:
                model.accumulate(rows[32:64])
                self.assertEqual(rebuild.call_count, 1)

    def test_fractional_resident_and_closing_handover(self):
        rows, initial = _data()
        for engine in ("torch", "faiss"):
            model = EmeansIVF(8, 8, **_options(engine))
            centroids, labels = model.train(rows, initial)
            self.assertEqual(centroids.shape, (8, 8))
            self.assertTrue((labels >= 0).all())
            distance, ids = model.assign(rows)
            exact = (rows[:, None] - centroids[None]).square().sum(-1)
            torch.testing.assert_close(distance, exact.min(1).values)
            torch.testing.assert_close(ids, exact.argmin(1))
            # Duplicated initial centroids cause moves in the sole fractional
            # pass; resident training must then reassign stale captured rows.
            short = EmeansIVF(8, 8, **dict(_options(engine), nepochs=.6, batch_size=8))
            result, labels = short.train(rows, torch.zeros_like(initial))
            self.assertGreater(short.n_splits, 0)
            self.assertIsNotNone(short.closing_handover_stats)
            self.assertEqual(int((labels >= 0).sum()), 96)
            self.assertTrue(torch.isfinite(result).all())

    def test_sse_with_revival_disabled_and_no_captured_labels(self):
        rows, initial = _data()
        for threshold in (0, .1):
            model = EmeansIVF(8, 8, **dict(_options(), unused_threshold=threshold,
                                         capture_assignments=False))
            model.begin(len(rows), initial)
            stats = model.accumulate(rows[:32])
            expected = (rows[:32, None] - initial[None]).square().sum(-1).min(1).values.double().sum()
            torch.testing.assert_close(stats["objective"], expected)
            self.assertNotIn("labels", stats)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
    def test_cpu_gloo_ownership_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(_gloo_worker,
                args=(2, "file://" + directory + "/init"), nprocs=2, start_method="spawn")

    @unittest.skipUnless(torch.cuda.is_available() and dist.is_nccl_available(), "requires CUDA/NCCL")
    def test_local_cuda_without_group_and_one_owner_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(_cuda_local_worker,
                args=("file://" + directory + "/init",), nprocs=1, start_method="spawn")


if __name__ == "__main__":
    unittest.main()
